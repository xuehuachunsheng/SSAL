from abc import ABC, abstractmethod

import json
import os,sys
import torch
import numpy as np
from PIL import Image
from scipy.special import softmax

# Batch Sample Selection Strategy for DeepAL
class BatchSampleSelectionStrategy(ABC):
    def __init__(self, U, classifier, q_budget, DEVICE) -> None:
        self.U = U # unlabeled pool
        self.classifier = classifier # classifier without softmax
        self.q_budget = q_budget
        self.device = DEVICE
        
    @abstractmethod
    def select(self): # Return the selected sample idx
        pass

    # Shift sample idd from U --> L
    def update(L, U, sample_idx:list):
        for idd in sample_idx:
            assert idd in U.data
            assert idd not in L.data
            L.data[idd] = U.data[idd]
            del U.data[idd]
        L.dataidx = list(L.data)
        U.dataidx = list(U.data)

class SelfSupervisedAL(BatchSampleSelectionStrategy):
    '''
    dc: cutoff distance
    it is the ratio of the maximum distance among sample features
    classifier recieve features but not original input
    L: labeled set
    U: unlabeled pool
    C: The number of classes
    feature_extractor: feature extractor 
    classifier: linear classifier, the last FC layer of the CNN
    q_budget: query budget of each AL process
    DEVICE: cpu or GPU
    dc: cutoff distance
    info_metric: value in ["U", "M", "F"], U: uncertainty, M: mastery, F: fusion
    balance: Whether maintains class balance or not.
    '''
    def __init__(self, 
                 L, 
                 U, 
                 C, 
                 feature_extractor, 
                 classifier, 
                 q_budget, 
                 DEVICE, 
                 dc=0.05,
                 info_metric="F",
                 balance=True) -> None:
        super().__init__(U, classifier, q_budget, DEVICE)
        self.L = L
        self.C = C # 类别数量
        self.dc = dc
        self.feature_extractor = feature_extractor
        self.im_features = None
        self.dist_mat = None
        self.rho = None
        self.delta = None
        
        # 信息度量方法，用于消融实验
        assert info_metric in ["U", "M", "F"]
        self.info_metric = info_metric
        self.balance = balance
        
        # 存储状态
        self.states = []

    def select_balance(self):
        idx = self.U.dataidx
        # 将样本信息按类别添加
        ssal_scores = [{} for _ in range(self.C)]
        for id in idx:
            c_logit = self.logits[id]
            current_c = np.argmax(c_logit)
            probs = softmax(c_logit)
            unc = -np.sum(probs * np.log(probs)).item()
            ssal_scores[current_c][id] = (self.info[id], # info
                                          unc,
                                          self.rho[id],  # rho
                                          self.delta[id]) # delta
            
        print("\nCompute sorted ssal scores by I_umf...")
        # Step 2. Ranking by I_umf descendant
        sorted_ssal_scores = []
        for c_scores in ssal_scores:
            if len(c_scores) == 0:
                sorted_ssal_scores.append([])
                continue
            s_scores = sorted(list(c_scores.items()), key=lambda x: -x[1][0])
            sorted_ssal_scores.append(s_scores)
        print("sorted_ssal_score length: ", [len(s_score) for s_score in sorted_ssal_scores])    
        print("Selecting samples...")
        # Note: Ensure two issues: 
        # (1) The number of queried samples of each class is enough
        # (2) The budget is satisfied predefined q_budget
        assert self.q_budget < len(self.U.dataidx)
        selected_CC = [[] for _ in range(len(self.nc))]
        rem_idx = [[] for _ in range(len(self.nc))]
        n_selected = 0
        for i, c_scores in enumerate(sorted_ssal_scores):
            if len(c_scores) <= self.bc[i]:
                for ele in c_scores:
                   selected_CC[i].append(ele[0]) # 全部选择
                   n_selected += 1
                   if n_selected % 500 == 0:
                       print("\r{}".format(n_selected), end="")
                continue
            
            for j,ele in enumerate(c_scores):
                if len(selected_CC[i]) >= self.bc[i]:
                    for t_ele in c_scores[j+1:]:
                        rem_idx[i].append(t_ele[0])    
                    break
                selected_CC[i].append(ele[0])
                n_selected += 1
                if n_selected % 500 == 0:
                    print("\r{}".format(n_selected), end="")
        num_rem_idx = sum([len(j) for j in rem_idx])
        
        print("\nRemaining idx length: ", num_rem_idx)
        print("\nSupply the remaining indices if n_selected < self.q_budget...")
        
        assert n_selected <= self.q_budget
        c_id = 0 # 第几个类
        cc_idx = [0] * len(self.nc) # 第几个类当前挑选的下标
        n_rem = 0
        while n_selected < self.q_budget and n_rem < num_rem_idx:
            if cc_idx[c_id] < len(rem_idx[c_id]):
                selected_CC[c_id].append(rem_idx[c_id][cc_idx[c_id]])
                cc_idx[c_id] += 1
                n_selected += 1
                n_rem += 1
            c_id = (c_id + 1) % len(self.nc)
        select_idx = []
        #print(selected_CC)
        for cc in selected_CC:
            select_idx.extend(cc)
        print("Final Selecting Samples: ", len(select_idx))
        return select_idx
        
    def select(self):
        print("********Extracting features*********\n")
        self.extract_features()
        print("\n********Computing Logits*********\n")
        self.compute_logits()
        # print("********Computing Distance Matrix*********\n")
        # self.compute_distmat()
        # print("********Computing Rho*********\n")
        # self.compute_rho()
        # print("********Computing Delta*********\n")
        # self.compute_delta()
        print("\n********Computing Rho and Delta*********\n")
        self.compute_rhodelta()
        print("********Computing SampleInformation*********\n")
        self.compute_info()
        print("********Computing Number of Labeled Samples*********\n")
        self.compute_nc()
        print("\n", self.nc)
        print("********Optimal Budget for Each Class*********")
        self.compute_bc()
        print(self.bc)
        
        if self.balance:
            # 考虑类别平衡的样本选择
            self.select_idx = self.select_balance()
        else:
            # 直接根据信息量选择样本
            info_sorted = sorted(self.info.items(), key=lambda x: -x[1])
            self.select_idx = [x[0] for x in info_sorted[:self.q_budget]]
            
    # 在执行select之后，update之前，存储当前的状态
    # 当前轮数
    # 包括前一轮的n_c；
    # 当前轮的最优b_c；
    # 当前轮所查询的所有样本的id
    # 当前轮所有样本的id
    # 当前轮所有样本的特征 -- 便于可视化 得单独存储
    # 当前轮所有样本的Rho和Delta值
    # 当前轮所有样本的logits -- 便于计算不确定性
    # 格式：
    # [
    # {
    # "queryid":xxx, 
    # "nc": nc_vector, 
    # "bc": bc_vector, 
    # "pool_ids": [],
    # "select_ids": [],
    # "rho": {id1:rho1,...},
    # "delta": {id1:delta1,...},
    # "entropy": {id1:entropy1,...}
    # "logits": logits_file_path,
    # "features": features_file_path,
    # }
    # ]
    def store_states(self, states_file_path):
        assert states_file_path is not None and isinstance(states_file_path, str)
        assert os.path.exists(os.path.dirname(states_file_path))
        states = self.states
        # try:
        #     f = open(states_file_path, "r")
        #     states = json.load(f)
        #     f.close()
        # except:
        #     states = []
        queryid = 1
        if len(states) > 0:
            queryid = states[-1]["queryid"] + 1 
        c_state = {}
        c_state["queryid"] = queryid
        c_state["nc"] = self.nc.tolist()
        c_state["bc"] = self.bc.tolist()
        c_state["pool_ids"] = self.U.dataidx
        c_state["select_ids"] = self.select_idx
        c_state["rho"] = self.rho
        c_state["delta"] = self.delta
        entropy = {}
        idx = self.U.dataidx
        for id in idx:
            probs = self.logits[id]
            probs = softmax(probs)
            ent = -np.sum(probs * np.log(probs))
            # ent is a np.float32 type, convert it to naive float
            entropy[id] = ent.item()
        c_state["entropy"] = entropy
        
        # 样本的logits单独存储
        logits = {}
        for id in self.logits:
            c_logit = self.logits[id].tolist()
            logits[id] = c_logit
        dir_path = os.path.dirname(states_file_path)
        logits_file_path = os.path.join(dir_path, "logits.{}.json".format(queryid))
        with open(logits_file_path, "w") as f:
            json.dump(logits, f)
        c_state["logits"] = logits_file_path
        
        # 样本特征单独存储
        im_features = {}
        for id in self.im_features:
            c_features = self.im_features[id].tolist()
            im_features[id] = c_features
        fea_file_path = os.path.join(dir_path, "features.{}.json".format(queryid))
        with open(fea_file_path, "w") as f:
            json.dump(im_features, f)
        c_state["features"] = fea_file_path
        
        self.states.append(c_state)
        with open(states_file_path, "w") as f:
            json.dump(self.states, f)
        
    # 计算labeled set中每个类别的样本数量
    def compute_nc(self):
        nc = np.zeros(self.C, dtype=np.int32)
        assert len(self.L.dataidx) == len(self.L.data)
        for i, id in enumerate(self.L.dataidx):
            label,_ = self.L.data[id]
            nc[label] += 1
            if (i+1) % 2000 == 0:
                print("\r Ldata:{}/{}".format(i, len(self.L.dataidx)), end="")
        self.nc = nc
    
    def compute_bc(self):
        nc = np.asarray(self.nc)
        C = self.C
        mean_nc = np.mean(nc)
        C_hat = np.where(nc - mean_nc < self.q_budget / C)[0] # C_hat
        assert len(C_hat) != 0
        bc = np.zeros(C, dtype=np.int32)
        for i in range(C): 
            if nc[i] - mean_nc < self.q_budget / C:
                bc[i] = np.round(mean_nc - nc[i] + self.q_budget / len(C_hat))
        
        # 严格控制查询预算
        t_Bi = int(np.sum(bc))
        while t_Bi < self.q_budget: # 如果实际小于预期budget，则随机选择C_hat中的类别提高其查询数量，直到满足当前budget
            tc = np.random.randint(low=0, high=len(C_hat))
            bc[C_hat[tc]] += 1
            t_Bi += 1
        while t_Bi > self.q_budget: # # 如果实际大于预期budget，则随机选择C_hat中的类别降低其查询数量，直到满足当前budget
            tc = np.random.randint(low=0, high=len(C_hat))
            if bc[C_hat[tc]] > 0: # 不允许bc小于0
                bc[C_hat[tc]] -= 1
                t_Bi -= 1
        assert t_Bi == self.q_budget
        self.bc = bc
    
    def compute_info(self):
        idx = self.U.dataidx
        info = {}
        for id in idx:
            probs = self.logits[id]
            probs = softmax(probs)
            unc = -np.sum(probs * np.log(probs))
            rho = self.rho[id]
            delta = self.delta[id]
            if self.info_metric == "F":
                info[id] = unc * rho * delta
            elif self.info_metric == "U":
                info[id] = unc
            elif self.info_metric == "M":
                info[id] = rho * delta
            else:
                raise Exception("This info metric is unimplemented now..")
        self.info = info
    
    def extract_features(self):
        idx = self.U.dataidx
        im_features = {}
        self.feature_extractor.eval()
        with torch.no_grad():
            for count,i in enumerate(idx):
                label,img = self.U.data[i]
                img = Image.open(img) if isinstance(img, str) else Image.fromarray(img)
                img = img.convert("RGB")
                img = self.U.transform(img).float()
                img = img.to(self.device)
                feature = self.feature_extractor.get_features(img[None, ...])[0].cpu().numpy()
                im_features[i] = feature
                if (count + 1) % 2000 == 0:
                    print("\rU count:{}/total:{}".format(count,len(idx)), end="")
        self.im_features = im_features

    def compute_logits(self):
        idx = self.U.dataidx
        logits = {}
        self.classifier.eval()
        with torch.no_grad():
            for count, i in enumerate(idx):
                im_feature = torch.tensor(self.im_features[i]).to(self.device)
                logit = self.classifier(im_feature[None,...])[0].cpu().numpy()
                logits[i] = logit
                if (count + 1) % 2000 == 0:
                    print("\rU count:{}/total:{}".format(count,len(idx)), end="")
        self.logits = logits
    
    # 利用分块的方式存储距离矩阵，直接计算rho和delta
    def compute_rhodelta(self):
        idx = self.U.dataidx
        n = len(idx)
        d = len(self.im_features[idx[0]])
        # Step 1. 将特征矩阵放入显存
        feature_mat = np.empty((n,d))
        for i, id in enumerate(idx):
            feature_mat[i] = self.im_features[id]
        f_mat = torch.tensor(feature_mat).to(self.device)
        
        # to According to GPU memeroy size
        GAP = 2000
        # Step 1. 计算真实的cutoff distance dc
        print("Computing cutoff distance dc....., GAP = ", GAP)
        max_dist = 0
        for i in range(0,n,GAP): 
            start = i
            end = i + GAP if i + GAP < n else n
            part_distmat = torch.cdist(f_mat[start:end], f_mat) 
            c_max_dist = torch.max(part_distmat).data.item()
            if c_max_dist > max_dist:
                max_dist = c_max_dist
            del part_distmat
            print("\r{}/{}".format(i,n), end="", flush=True)
        real_dc = self.dc * max_dist
        print("\nd_c ratio: {}, d_c: {}".format(self.dc,real_dc))
        torch.cuda.empty_cache()
        
        # Step 2. 计算rho
        print("Computing Rho....")
        rho = {}
        for i in range(0,n,GAP): 
            start = i
            end = i + GAP if i + GAP < n else n
            part_distmat = torch.cdist(f_mat[start:end], f_mat)
            for j in range(start, end):
                dist_vec = part_distmat[j-start]
                rho[idx[j]] = torch.sum(torch.where(dist_vec<real_dc,1,0)).data.item()
                if (j+1) % 2000 == 0:
                    print("\rRho: {}/{}".format(j+1, len(idx)),end="",flush=True) 
            del part_distmat
            torch.cuda.empty_cache()
        print("\nAverage Rho: ", np.sum(list(rho.values())) / len(rho))
        print("Max Rho: ", np.max(list(rho.values())))
        print("Min Rho: ", np.min(list(rho.values())))
        
        # Step 2. 计算delta            
        print("Computing Delta....")
        delta = {}
        for i in range(0,n,GAP): 
            start = i
            end = i + GAP if i + GAP < n else n
            part_distmat = torch.cdist(f_mat[start:end], f_mat)
            for j in range(start, end):
                c_id = idx[j]
                dist_vec = part_distmat[j-start]
                c_rho = rho[c_id]
                dist_sort_indx = torch.argsort(dist_vec).cpu().numpy().astype(np.int32)
                for ind in dist_sort_indx:
                    true_sample_id = idx[ind]
                    if rho[true_sample_id] > c_rho:
                        delta[c_id] = dist_vec[ind].data.item()
                        break
                if c_id not in delta:
                    delta[c_id] = torch.max(dist_vec).data.item()
                if (j+1)%2000 == 0:
                    print("\rDelta: {}/{}".format(j+1, len(idx)),end="",flush=True)    
            del part_distmat
            torch.cuda.empty_cache()
                
        print("\nAverage Delta: ", np.sum(list(delta.values())) / len(delta))
        print("Max Delta: ", np.max(list(delta.values())))
        print("Min Delta: ", np.min(list(delta.values())))
        
        del f_mat
        torch.cuda.empty_cache()
        self.rho = rho
        self.delta = delta
    
    # 计算距离矩阵
    # 已过时
    def compute_distmat(self):
        idx = self.U.dataidx
        n = len(idx)
        d = len(self.im_features[idx[0]])
        feature_mat = np.empty((n,d))
        for i, id in enumerate(idx):
            feature_mat[i] = self.im_features[id]
            
        # 放入显存再计算距离
        # 可能引起显存溢出问题
        f_mat = torch.tensor(feature_mat).to(self.device)
        dist_mat = torch.cdist(f_mat, f_mat, p=1)
        self.dist_mat = dist_mat
        
        print("Distance matrix dimension: ", self.dist_mat.shape)
    
    # 也就是文中的gamma
    # 已过时
    def compute_rho(self):
        idx = self.U.dataidx
        rho = {}
        max_dist = torch.max(self.dist_mat).data.item()
        real_dc = self.dc * max_dist
        for i, id in enumerate(idx):
            dist_vec = self.dist_mat[i]
            rho[id] = torch.sum(torch.where(dist_vec<real_dc,1,0)).data.item()
            
            if (i+1)%100 == 0:
                print("\rRho: {}/{}".format(i, len(idx)),end="",flush=True)    
        self.rho = rho
        
        print("Average rho value: ", np.sum(self.rho.values()) / len(self.rho))
    
    # 计算minimum distance
    # 已过时
    def compute_delta(self):
        idx = self.U.dataidx
        delta = {}
        for i, id in enumerate(idx):
            # 当前样本(id)的距离向量
            dist_vec = self.dist_mat[i]
            # 当前样本的密度
            rho = self.rho[id]
            # 获得距离向量参数排序下标
            dist_sort_indx = torch.argsort(dist_vec).cpu().numpy().astype(np.int32)
            for ind in dist_sort_indx:
                true_sample_id = idx[ind]
                if self.rho[true_sample_id] > rho:
                    delta[id] = dist_vec[ind]
                    break
                
            if id not in delta:
                delta[id] = torch.max(dist_vec).data.item()
                
            if (i+1)%100 == 0:
                print("\rDelta: {}/{}".format(i, len(idx)),end="",flush=True)    
        self.delta = delta
        
    def update_data(self):
        L,U = self.L,self.U
        for idd in self.select_idx:
            assert idd in U.data
            assert idd not in L.data
            L.data[idd] = U.data[idd]
            del U.data[idd]
        L.dataidx = list(L.data)
        U.dataidx = list(U.data)
    
