# -*- coding: utf-8 -*-

#requestment module

'''
solve downlink single user PA problem
'''

import os
import time
import scipy
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import pickle
from PA_alg import PA_alg
from scipy import special, io

dtype = np.float32
flag_fig = True


#parameters
# =============================================================================
#Small scale fading (Jakes model)
fd = 10 #10Hz
Ts = 20e-3 #20ms
Ns = 5e1 #5e1 = 50, There are Ns time slots per episode


#Cellular network
n_x = 5 #蜂巢網路橫向個數
n_y = 5 #蜂巢網路縱向個數
N = n_x * n_y # BS number
L = 2 #六邊形蜂巢網路往外2層
C = 16 # adjascent BS (Top 16)
c = 3*L*(L+1) + 1 # adjascent BS, including itself


#User 
meanM = 6   # lamda: average user number in one BS
minM = 6   # maximum user number in one BS
maxM = 6   # maximum user number in one BS
K = maxM * c # maximum adjascent users, including itself #76
M = maxM * N # maximum users in this cellular network #100 
min_dis = 0.01 #km, distance between user and BS
max_dis = 1. #km, distance between user and BS  /基站半徑


#Power
min_p = 5. #dBm, emitting power constraints
max_p = 38. #dBm, emitting power constraints
maxP = 1e-3*pow(10., max_p/10.) #w
p_n = -114. #dBm, AWGN power
sigma2_ = 1e-3*pow(10., p_n/10.) #w

power_num = 10 #action_num, power level number, abs(A)
state_num = 3*C + 2    #  3*K - 1  3*C + 2 
power_set = np.hstack([np.zeros((1), dtype=dtype), 1e-3*pow(10., np.linspace(min_p, max_p, power_num-1)/10.)]) #w Power選項

#DQN model
OBSERVE = 100
EPISODE = 10000  # 訓練週期(循環次數)
train_interval = 10 # learn 間隔,請小於Ns且盡量為他的因數
target_replace_iter = 500 # target network 更新間隔
batch_size = 256 #每次訓練筆數

TEST_EPISODE = 500 #測試週期(循環次數)
memory_size = M * 400

INITIAL_EPSILON = 0.2  #epsilon-greedy
FINAL_EPSILON = 0.0001 #epsilon-greedy

learning_rate = 0.001  #in Adam 
gamma = 0. #有多關注未來(權重)

#Else
W_ = torch.ones(M) #1*100


file_path = os.getcwd() + '\save_state_dict_1115_55_66_3.txt'

# =============================================================================


#數學架構 
# ============================================================================= 
# 通道係數
def Generate_H_set():
 
    H_set = np.zeros([M,K,int(Ns)], dtype=dtype)
    pho = np.float32(scipy.special.k0(2*np.pi*fd*Ts))
    H_set[:,:,0] = np.kron(np.sqrt(0.5*(np.random.randn(M, c)**2+np.random.randn(M, c)**2)), np.ones((1,maxM), dtype=np.int32))
    for i in range(1,int(Ns)):
        H_set[:,:,i] = H_set[:,:,i-1]*pho + np.sqrt((1.-pho**2)*0.5*(np.random.randn(M, K)**2+np.random.randn(M, K)**2))
        
    
    path_loss = Generate_path_loss()
    H2_set = np.square(H_set) * np.tile(np.expand_dims(path_loss, axis=2), [1,1,int(Ns)]) 
    H2_set = torch.from_numpy(H2_set)
    return H2_set

#通道係數 : H2_set = H^2 * B = (small scale fading)^2 * (large scale fading)
# H2_set (100, 76, 50)

    
def Generate_environment():
    # path_matrix中間部分依序放入 0 ~ 100(M)
    path_matrix = M*np.ones((n_y + 2*L, n_x + 2*L, maxM), dtype = np.int32)
    for i in range(L, n_y+L):
        for j in range(L, n_x+L):
            for l in range(maxM):
                path_matrix[i,j,l] = ((i-L)*n_x + (j-L))*maxM + l
    
    
    p_array = np.zeros((M, K), dtype = np.int32)    
    for n in range(N):
        #基地站 位置i, j
        i = n//n_x 
        j = n%n_x
        Jx = np.zeros((0), dtype = np.int32)
        Jy = np.zeros((0), dtype = np.int32)
        for u in range(i-L, i+L+1):
            v = 2*L+1-np.abs(u-i)
            jx = j - (v-i%2)//2 + np.linspace(0, v-1, num = v, dtype = np.int32) + L
            jy = np.ones((v), dtype = np.int32)*u + L
            Jx = np.hstack((Jx, jx))
            Jy = np.hstack((Jy, jy))
        for l in range(maxM):
            for k in range(c):
                for u in range(maxM):
                    p_array[n*maxM+l,k*maxM+u] = path_matrix[Jy[k],Jx[k],u]
    p_main = p_array[:,(c-1)//2*maxM:(c+1)//2*maxM]
    for n in range(N):
        for l in range(maxM):
            temp = p_main[n*maxM+l,l]
            p_main[n*maxM+l,l] = p_main[n*maxM+l,0]
            p_main[n*maxM+l,0] = temp
    p_inter = np.hstack([p_array[:,:(c-1)//2*maxM], p_array[:,(c+1)//2*maxM:]])
    p_array =  np.hstack([p_main, p_inter])             
    
    #人數分布採用 poisson 分布  (可調整至其他分布), 0 < user <4
    user = np.maximum(np.minimum(np.random.poisson(meanM, (N)), maxM), minM)
    user_list = np.zeros((N, maxM), dtype = np.int32)
    for i in range(N):
        user_list[i,:user[i]] = 1
    
    #等於 0 的狀況下(無使用者)找出編號改成M，其他照舊    
    for k in range(N):
        for i in range(maxM):
            
            if user_list[k,i] == 0.:
                p_array = np.where(p_array == k*maxM+i, M, p_array) #满足條件，输出 M，不满足输出 p_array(照舊)
                # 前四行有人才有數字(使用者編號)，無人就給100(M)
    
    #從 p_array 製作 p_list                
    p_list = list()
    for i in range(M):
        p_list_temp = list() 
        for j in range(K):
            p_list_temp.append([p_array[i,j]])
        p_list.append(p_list_temp)               
    
    
    return p_array, p_list, user_list


#user_list : 基地站與使用者分布狀況 25*4
#p_array : 使用者影響狀況編號表格
#p_list : 使用者影響況編號清單
    
    

def Generate_path_loss():
    slope = 0.      #0.3
    p_tx = np.zeros((n_y, n_x))
    p_ty = np.zeros((n_y, n_x))
    p_rx = np.zeros((n_y, n_x, maxM))
    p_ry = np.zeros((n_y, n_x, maxM))   
    dis_rx = np.random.uniform(min_dis, max_dis, size = (n_y, n_x, maxM))
    phi_rx = np.random.uniform(-np.pi, np.pi, size = (n_y, n_x, maxM))    
    for i in range(n_y):
        for j in range(n_x):
            p_tx[i,j] = 2*max_dis*j + (i%2)*max_dis
            p_ty[i,j] = np.sqrt(3.)*max_dis*i
            for k in range(maxM):  
                p_rx[i,j,k] = p_tx[i,j] + dis_rx[i,j,k]*np.cos(phi_rx[i,j,k])
                p_ry[i,j,k] = p_ty[i,j] + dis_rx[i,j,k]*np.sin(phi_rx[i,j,k])
    dis = 1e10 * np.ones((M, K), dtype = dtype)
    lognormal = np.zeros((M, K), dtype = dtype)
    for k in range(N):
        for l in range(maxM):
            for i in range(c):
                for j in range(maxM):
                    if p_array[k*maxM+l,i*maxM+j] < M: #<100
                        bs = p_array[k*maxM+l,i*maxM+j]//maxM                        
                        dx2 = np.square((p_rx[k//n_x][k%n_x][l]-p_tx[bs//n_x][bs%n_x]))
                        dy2 = np.square((p_ry[k//n_x][k%n_x][l]-p_ty[bs//n_x][bs%n_x]))
                        distance = np.sqrt(dx2 + dy2) #開根號
                        dis[k*maxM+l,i*maxM+j] = distance
                        std = 8. + slope * (distance - min_dis)
                        lognormal[k*maxM+l,i*maxM+j] = np.random.lognormal(sigma = std)
    
    
    path_loss = lognormal*pow(10., -(120.9 + 37.6*np.log10(dis))/10.) #large scale fading
    return path_loss

#path_loss : Large scale fading, 給Generate_H_set使用
  
#公式(佔位符)
def Calculate_rate(P, H2, sigma2, W):
    # maxC = 1000.
    P_extend = torch.cat((P,torch.zeros(1)),0)# P 尾巴多加一個 0 ,shape = (M+1, )

    P_matrix = torch.take(P_extend, torch.from_numpy(p_array).long()) #p_array 變成 Power_matrix
    path_main = np.multiply(H2[:,0], P_matrix[:,0]) #訊號 = power * path loss
    path_inter = torch.sum(torch.multiply(H2[:,1:], P_matrix[:,1:]), axis=1) #干擾
    sinr = path_main / (path_inter + sigma2) #訊噪雜訊比
    # sinr = torch.minimum(sinr, maxC)       #capped sinr
    rate = W * np.log(1. + sinr)/np.log(2) #C^t in paper 
    # rate = W * np.log2(1. + sinr) #C^t in paper 
    rate_extend = torch.cat((rate, torch.zeros(1)),0)
    rate_matrix = torch.take(rate_extend, torch.from_numpy(p_array).long())
    
    sinr_norm_inv = H2[:,1:] / H2[:,0:1].repeat(1,K-1) #訊噪雜訊比
    sinr_norm_inv = torch.log(1. + sinr_norm_inv)/np.log(2)# log representation  C(t, n, k)
    reward = torch.sum(rate) #we want to max this one using DQN
    return rate_matrix, sinr_norm_inv, P_matrix, reward

# rate_matrix = (100, 76) C^t in paper 矩陣
# sinr_norm_inv = (100, 75)
# P_matrix = (100, 76) 功率矩陣
# reward = () 獎勵函數，單值


# 建立環境狀態 Matrix
def Generate_state(rate_last, p_last, sinr_norm_inv):
    '''
    Generate state matrix
    ranking
    state including:
    1.rate[t-1]          [M,K]  rate_last
    2.power[t-1]         [M,K]  p_last
    3.sinr_norm_inv[t]   [M,K-1]  sinr_norm_inv
    '''
#    s_t = np.hstack([rate_last])
#    s_t = np.hstack([rate_last, sinr_norm_inv])

    indices1 = np.tile(np.expand_dims(np.linspace(0, M-1, num=M, dtype=np.int32), axis=1),[1,C])
    indices2 = np.argsort(sinr_norm_inv, axis = 1)[:,-C:]
    rate_last = np.hstack([rate_last[:,0:1], rate_last[indices1, indices2+1]])
    p_last = np.hstack([p_last[:,0:1], p_last[indices1, indices2+1]])
    sinr_norm_inv = sinr_norm_inv[indices1, indices2]
    s_t = np.hstack([rate_last, p_last, sinr_norm_inv])
    s_t = torch.from_numpy(s_t)
    return s_t
# s_t = 100*50
# rate_last (前一個時刻的C^t)= 100*17, p_last (前一個時刻的power) = 100*17, sinr_norm_inv(干擾源) = 100*16

#基站與使用者分布狀況圖
def Plot_environment():
    if flag_fig:

        plt.close('all')
        p_tx = np.zeros((n_y, n_x))
        p_ty = np.zeros((n_y, n_x))
        p_rx = np.zeros((n_y, n_x, maxM))
        p_ry = np.zeros((n_y, n_x, maxM))   
        dis_r = np.random.uniform(min_dis, max_dis, size = (n_y, n_x, maxM))
        phi_r = np.random.uniform(-np.pi, np.pi, size = (n_y, n_x, maxM))    
        for i in range(n_y):
            for j in range(n_x):
                p_tx[i,j] = 2*max_dis*j + (i%2)*max_dis
                p_ty[i,j] = np.sqrt(3.)*max_dis*i
                for k in range(maxM):  
                    p_rx[i,j,k] = p_tx[i,j] + dis_r[i,j,k]*np.cos(phi_r[i,j,k])
                    p_ry[i,j,k] = p_ty[i,j] + dis_r[i,j,k]*np.sin(phi_r[i,j,k])
                    
        plt.close('all')
        plt.figure(1)

        for j in range(n_x):
            for i in range(n_y):
                for k in range(6):
                    x_t = [p_tx[i,j]+2/np.sqrt(3.)*max_dis*np.sin(np.pi/3*k), p_tx[i,j]+2/np.sqrt(3.)*max_dis*np.sin(np.pi/3*(k+1))]
                    y_t = [p_ty[i,j]+2/np.sqrt(3.)*max_dis*np.cos(np.pi/3*k), p_ty[i,j]+2/np.sqrt(3.)*max_dis*np.cos(np.pi/3*(k+1))]
                    plt.plot(x_t, y_t, color="black")
        for i in range(n_y):
            for j in range(n_x):
                for l in range(maxM):
                    if user_list[i*n_x+j,l]:      
                        rx = plt.scatter(p_rx[i,j,l], p_ry[i,j,l], marker='o', label='2', s=25, color='orange')
                        plt.text(p_rx[i,j,l]+0.03, p_ry[i,j,l]+0.03, '%d' %((i*n_x+j)*maxM+l), ha ='center', va = 'bottom', fontsize=12)
        tx = plt.scatter(p_tx, p_ty, marker='x', label='1', s=45)
        for i in range(n_y):
            for j in range(n_x):
                plt.text(p_tx[i,j]+0.1, p_ty[i,j]+0.1, '%d' %(i*n_x+j), ha ='center', va = 'bottom', fontsize=15, color = 'r')
        plt.legend([tx, rx], ["BS", "User"])
        plt.xlabel('X axis (km)')
        plt.ylabel('Y axis (km)')
        plt.axis('equal')
        plt.savefig("Plot_environment.png")
        plt.show()
        
        



# =============================================================================
# 初始化 狀態
def Initial_para():
    H2_set = Generate_H_set()
    s_next, _ = Step(torch.zeros([M]), H2_set[:,:,0])

    return H2_set, s_next  

def Step(p_t, H2_t): #p_t.shape = (M, )  H2_t.shape = (M, K) 

    rate_last, sinr_norm_, p_last, reward_= Calculate_rate(P = p_t, H2 = H2_t, W = W_, sigma2 = sigma2_)
    s_next = Generate_state(rate_last, p_last, sinr_norm_)
    
    return s_next, reward_  


# Q_Table 
'''
把 state 傳入後，得出每個 action 的分數，分數越高的 action 越有機會被挑選。
而我們的目標是在當前 state 下，讓對未來越有利的 action 分數能越高。 
Q_Table , Q_Table , Q_Table 
'''
class Net(nn.Module):
    def __init__(self, ):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(state_num, 128)
        # self.fc1.weight.data.normal_(0, 0.1)   # initialization
        self.fc2 = nn.Linear(128, 64)
        # self.fc2.weight.data.normal_(0, 0.1)   # initialization        
        self.out = nn.Linear(64, power_num)
        # self.out.weight.data.normal_(0, 0.1)   # initialization

    def forward(self, x):
        x = x.float()
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        x = F.relu(x)
        actions_value = self.out(x)
        

        return actions_value


net = Net()

    
class DQN(object):
    def __init__(self, state_num, power_num, batch_size, learning_rate, INITIAL_EPSILON, gamma, target_replace_iter, memory_size, EPISODE):
        self.eval_net, self.target_net = net, net

        self.memory = torch.zeros((memory_size, state_num * 2 + 2)) # 每個 memory 中的 experience 大小為 (state + reward + action + next_state)
        self.optimizer = torch.optim.Adam(self.eval_net.parameters(), lr=learning_rate)
        self.loss_func = nn.MSELoss()
        self.memory_counter = 0
        self.learn_step_counter = 0 # 讓 target network 知道什麼時候要更新
        self.state_num = state_num
        self.power_num = power_num
        self.batch_size = batch_size
        self.EPISODE = EPISODE
        self.learning_rate = learning_rate
        self.INITIAL_EPSILON = INITIAL_EPSILON
        self.gamma = gamma
        self.target_replace_iter = target_replace_iter
        self.memory_size = memory_size

    # 選擇動作 
    '''
    會根據 epsilon-greedy policy 選擇 action。
    訓練過程中有 epsilon 的機率 agent 會選擇亂（隨機）走，如此才有機會學習到新經驗。
    q_hat_ 為 DQN 的 reward matrix
    '''
    def Select_action(self, s_t, episode):
        #episode 從 0.2 降至 0.0001
        if episode > OBSERVE:
            epsilon = INITIAL_EPSILON - (episode - OBSERVE) * (INITIAL_EPSILON - FINAL_EPSILON) / (EPISODE - OBSERVE) 
        elif episode <= OBSERVE:
            epsilon = INITIAL_EPSILON
        else:
            epsilon = 0.
            
        q_hat_ = self.eval_net.forward(s_t)    # q_hat_ = [M, power_num] 
        best_action = torch.argmax(q_hat_, axis = 1) #橫著比較"值"返回最大值的"索引號", shape = [100, ]進入DNN產生的動作 0~9
        random_index = np.array(np.random.uniform(size = (M)) < epsilon, dtype = np.int32) #選擇 0(best_action) or 1(random_action)
        random_action = np.random.randint(0, high = power_num, size = (M)) #產生隨機動作_index 0~9    
        action_set = np.vstack([best_action, random_action]) 
        power_index = action_set[random_index, range(M)] # power_index.shape = (M, ) 
        power = power_set[power_index] # W, power.shape = (M, )
        power = torch.from_numpy(power) #轉 tensor
        power_index = torch.from_numpy(power_index) #轉 tensor
        return power, power_index  
        #輸出 power and power_index, shape = (M,)

    # 暫存區
    '''
    把狀態儲存至 memory matrix [50000, 102]
    一次儲存 [100, 102]
    記憶體滿了會覆蓋舊的    
    '''
    def store_transition(self, state, action, reward, next_state):
        # 合併成 transition
        reward = reward * W_
        reward = reward.unsqueeze(1)
        action = action.unsqueeze(1)
        transition =torch.cat((state, action, reward, next_state), 1)
        # 存進 memory, 舊 memory 可能會被覆蓋
        index = self.memory_counter % self.memory_size
        self.memory[index : index + M, :] = transition
        self.memory_counter = self.memory_counter + M
        
        
    # DQN 更新邏輯
    '''
    從 memory 中取樣學習
    本案例 Gamma = 0    
    '''
    def learn(self):
        # 隨機取樣 batch_size 個 experience        
        # size = batch_size, range = 0 to memory_size-1
        sample_index = np.random.choice(self.memory_size, self.batch_size) #從memory_size，隨機選擇 batch_size 個
        b_memory = self.memory[sample_index, :]
        b_state = torch.FloatTensor(b_memory[:, :self.state_num])
        b_action = torch.Tensor(b_memory[:, self.state_num:self.state_num+1])
        b_reward = torch.FloatTensor(b_memory[:, self.state_num+1:self.state_num+2])
        b_next_state = torch.FloatTensor(b_memory[:, -self.state_num:])

        # 計算現有 eval net 和 target net 得出 Q value 的落差
        q_eval = self.eval_net(b_state).gather(1, b_action.type(torch.int64)) # 重新計算這些 experience 當下 eval net 所得出的 Q value
        q_next = self.target_net(b_next_state).detach() # detach 才不會訓練到 target net
        q_target = b_reward + self.gamma * q_next.max(1)[0].view(self.batch_size, 1) # 計算這些 experience 當下 target net 所得出的 Q value
        loss = self.loss_func(q_eval, q_target) #收斂至loss值，穩定

        # Backpropagation
        self.optimizer.zero_grad() #梯度歸零
        loss.backward()
        self.optimizer.step()

        # 每隔一定次數 (target_replace_iter), 更新 target net，即複製 eval net 到 target net
        self.learn_step_counter += 1
        if self.learn_step_counter % self.target_replace_iter == 0:
            self.target_net.load_state_dict(self.eval_net.state_dict())


    def save_state_dict(self, file_path):
        """"Save model.
        file_path: output path to save model
        """
        model = { 'eval_net': self.eval_net.state_dict(), 'target_net': self.target_net.state_dict() }
        with open(file_path, 'wb') as fout:
            pickle.dump(model, fout)

    def load_state_dict(self, file_path):
        """"Load model.
        file_path: input path to load model
        """
        with open(file_path, 'rb') as fout:
            model = pickle.load(fout)
        self.eval_net.load_state_dict(model['eval_net'])
        self.target_net.load_state_dict(model['target_net'])



def Train():
    st = time.time()
    dqn_hist = list() 
    for k in range(1, EPISODE+1):
        dqn_hist.append(Train_episode(k))
        if k % 100 == 0: 
            print("Episode(train):%d   DQN: %.3f   Time cost: %.2fs  loading... 經過 %d天，剩下 %d天  " 
                  %(k, np.mean(dqn_hist[-100:]), time.time()-st, k*100/EPISODE, 100-(k*100/EPISODE) ) )
            st = time.time()
            
    dqn.save_state_dict(file_path)

    return dqn_hist

# 單次 Train_episode 執行動作    
def Train_episode(episode): 
    reward_dqn_list = list()
    H2_set, s_t = Initial_para()  #  return H2_set, s_next    


    for step_index in range(int(Ns)):
        #選擇動作
        p_t, a_t = dqn.Select_action(s_t, episode) #episode <=Train (k)=10000, return p_t = power, a_t = power_index 
        s_next, r_ = Step(p_t, H2_set[:,:,step_index]) #做動作去更新狀態        
        dqn.store_transition(s_t, a_t, r_, s_next)
        if episode > OBSERVE:
            if step_index % train_interval == 0:
                dqn.learn()
        #更新狀態        
        s_t = s_next
        reward_dqn_list.append(r_)
    

    
    dqn_mean = sum(reward_dqn_list)/(Ns*M) # bps/Hz per link
    return dqn_mean


# 結果測試 隨機取樣    
def Test_one():
    dqn.load_state_dict(file_path)
    reward_dqn_list = list()
    reward_fp_list = list()
    reward_wmmse_list = list()
    reward_mp_list = list()
    reward_rp_list = list()
    H2_set, s_t = Initial_para()
    
    for step_index in range(int(Ns)):
        q_hat_ = dqn.target_net.forward(s_t) #[N, power_num]
        p_t = power_set[torch.argmax(q_hat_, axis = 1)] # W
        p_t = torch.from_numpy(p_t)
        s_next, r_ = Step(p_t, H2_set[:,:,step_index])
        s_t = s_next
        reward_dqn_list.append(r_/M)
        
        H2_set_ = H2_set.numpy()
        pa_alg_set.Load_data(H2_set_[:,:,step_index], p_array) #pa_alg_set is funciton
        fp_alg, wmmse_alg, mp_alg, rp_alg = pa_alg_set.Calculate()
        
        fp_alg = torch.from_numpy(fp_alg)
        wmmse_alg = torch.from_numpy(wmmse_alg)
        mp_alg = torch.from_numpy(mp_alg)
        rp_alg = torch.from_numpy(rp_alg)
        
        
        _, _, _, r_fp= Calculate_rate(P = fp_alg, H2 = H2_set[:,:,step_index], W = W_, sigma2 = sigma2_)
        _, _, _, r_wmmse= Calculate_rate(P = wmmse_alg, H2 = H2_set[:,:,step_index], W = W_, sigma2 = sigma2_)
        _, _, _, r_mp= Calculate_rate(P = mp_alg, H2 = H2_set[:,:,step_index], W = W_, sigma2 = sigma2_)
        _, _, _, r_rp= Calculate_rate(P = rp_alg, H2 = H2_set[:,:,step_index], W = W_, sigma2 = sigma2_)
        
        reward_fp_list.append(r_fp/M)
        reward_wmmse_list.append(r_wmmse/M)
        reward_mp_list.append(r_mp/M)
        reward_rp_list.append(r_rp/M)   
        
    if flag_fig:
        import matplotlib.pyplot as plt
        plt.figure(3)
        window = 11
        y = list()
        y.append(Smooth(np.array(reward_dqn_list), window))
        y.append(Smooth(np.array(reward_fp_list), window))
        y.append(Smooth(np.array(reward_wmmse_list), window))
        y.append(Smooth(np.array(reward_mp_list), window))
        y.append(Smooth(np.array(reward_rp_list), window))
        label=['DQN','FP','WMMSE','Maximal power','Random power']
        color = ['royalblue', 'orangered', 'lawngreen', 'gold', 'olive']
        linestyle = ['-', '--', '-.', ':', '--']
        p = list()
        for k in range(5):
            p_temp, = plt.plot(range(int(Ns)), y[k], color = color[k], linestyle = linestyle[k], label = label[k])
            p.append(p_temp)
        plt.legend(loc = 7)
        plt.xlabel('Time slot')
        plt.ylabel('Average rate (bps), C^t ')        
        plt.grid()
        plt.show()
        plt.savefig("Test_one.png")
        print("Test_one: DQN: %.2f  FP: %.2f  WMMSE: %.2f  MP: %.2f  RP: %.2f" 
          %(np.mean(reward_dqn_list), np.mean(reward_fp_list), np.mean(reward_wmmse_list), np.mean(reward_mp_list), np.mean(reward_rp_list)))


# Smooth
def Smooth(a, window):
    out0 = np.convolve(a, np.ones(window, dtype=np.int32),'valid')/window
    r = np.arange(1, window-1, 2)
    start = np.cumsum(a[:window-1])[::2]/r
    stop = (np.cumsum(a[:-window:-1])[::2]/r)[::-1]
    return np.concatenate((start, out0, stop))

def Test():
    dqn.load_state_dict(file_path)
    dqn_hist = list()
    fp_hist = list()
    wmmse_hist = list()
    mp_hist = list()
    rp_hist = list()
    for k in range(1, TEST_EPISODE+1):
        dqn_mean, fp_mean, wmmse_mean, mp_mean, rp_mean = Test_episode()
        dqn_hist.append(dqn_mean)
        fp_hist.append(fp_mean)
        wmmse_hist.append(wmmse_mean)
        mp_hist.append(mp_mean)
        rp_hist.append(rp_mean)
    print("Test: DQN: %.3f  FP: %.3f  WMMSE: %.3f  MP: %.3f  RP: %.3f" 
          %(np.mean(dqn_hist), np.mean(fp_hist), np.mean(wmmse_hist), np.mean(mp_hist), np.mean(rp_hist)))
    print("Test: DQN: %.2f, %.2f, %.2f, %.2f, %.2f" 
          %(np.mean(dqn_hist), np.mean(fp_hist), np.mean(wmmse_hist), np.mean(mp_hist), np.mean(rp_hist)))
    # return dqn_hist, fp_hist, wmmse_hist, mp_hist, rp_hist

# 單次 Test_episode 執行動作    
def Test_episode():
    '''
    1.DQN
    2.FP
    3.WMMSE
    4.Maximum Power
    5.Random Power
    '''
    reward_dqn_list = list()
    reward_fp_list = list()
    reward_wmmse_list = list()
    reward_mp_list = list()
    reward_rp_list = list()
    H2_set, s_t = Initial_para()
    for step_index in range(int(Ns)):
        q_hat_ = dqn.target_net.forward(s_t) #[N, power_num]
        p_t = power_set[torch.argmax(q_hat_, axis = 1)] # W
        p_t = torch.from_numpy(p_t)
        s_next, r_ = Step(p_t, H2_set[:,:,step_index])
        s_t = s_next
        reward_dqn_list.append(r_)
        
        H2_set_ = H2_set.numpy()
        pa_alg_set.Load_data(H2_set_[:,:,step_index], p_array) #pa_alg_set is funciton
        fp_alg, wmmse_alg, mp_alg, rp_alg = pa_alg_set.Calculate()
        
        fp_alg = torch.from_numpy(fp_alg)
        wmmse_alg = torch.from_numpy(wmmse_alg)
        mp_alg = torch.from_numpy(mp_alg)
        rp_alg = torch.from_numpy(rp_alg)
              
        _, _, _, r_fp= Calculate_rate(P = fp_alg, H2 = H2_set[:,:,step_index], W = W_, sigma2 = sigma2_)
        _, _, _, r_wmmse= Calculate_rate(P = wmmse_alg, H2 = H2_set[:,:,step_index], W = W_, sigma2 = sigma2_)
        _, _, _, r_mp= Calculate_rate(P = mp_alg, H2 = H2_set[:,:,step_index], W = W_, sigma2 = sigma2_)
        _, _, _, r_rp= Calculate_rate(P = rp_alg, H2 = H2_set[:,:,step_index], W = W_, sigma2 = sigma2_)
        
        reward_fp_list.append(r_fp)
        reward_wmmse_list.append(r_wmmse)
        reward_mp_list.append(r_mp)
        reward_rp_list.append(r_rp)   
     

    dqn_mean = sum(reward_dqn_list)/(Ns*M)
    fp_mean = sum(reward_fp_list)/(Ns*M)
    wmmse_mean = sum(reward_wmmse_list)/(Ns*M)
    mp_mean = sum(reward_mp_list)/(Ns*M)
    rp_mean = sum(reward_rp_list)/(Ns*M)
    return dqn_mean, fp_mean, wmmse_mean, mp_mean, rp_mean

#RUN~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# 模擬
# p_array, p_list, user_list = Generate_environment()
# # 作圖
# Plot_environment()
# # 建立 DQN 模型
# dqn = DQN(state_num, power_num, batch_size, learning_rate, INITIAL_EPSILON, gamma, target_replace_iter, memory_size, EPISODE)
# #  Training
# dqn_hist = Train()
# # 其他作法模擬
# pa_alg_set = PA_alg(M, K, maxP)
# # 比較
# Test_one()
# Test()

for i in range(10):

    
    p_array, p_list, user_list = Generate_environment()
    Plot_environment()
    dqn = DQN(state_num, power_num, batch_size, learning_rate, INITIAL_EPSILON, gamma, target_replace_iter, memory_size, EPISODE)
    pa_alg_set = PA_alg(M, K, maxP)
    Test_one()
    



