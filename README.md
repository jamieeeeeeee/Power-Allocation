# Power Allocation in Multi-user Cellular Networks With Deep Q Learning Approach

## 前言
隨著無線通信的飛速發展，網絡規模越來越大，基站（BS）的數量也急劇增加。功率和無線資源的優化分配問題變得尤為突出，不適當的分配方案會降低網絡頻譜效率，也會讓干擾增加，而部分算法也會因複雜性難以實現。我這次的目標就是利用機器學習中強化學習的方式來更新小型蜂巢基站的downlink power，把周圍干擾小區的SINR和Power當作環境，利用給定好的power大小當作動作，利用Q learning 演算法找到可以讓整體蜂巢網路的SINR達到最高的狀態，最後做出最佳策略的Q Table 也就是儲存好權種值的DNN model，幫助這個蜂巢基站找出整體SINR最大的狀況。這個系統可以使用在初期部屬以及低於臨界值需要自我調整的時候。會選擇DQN這個方法是因為他的評價策略相較容易，且此方法泛化性高，也不用對資料而外進行標註作業。

**這個實驗是基於 "Power Allocation in Multi-user Cellular Networks With Deep Q Learning Approach" 這篇論文實作**

## 資料模擬
* 基地站數量 : 3 * 3, 4 * 4, 5 * 5, 6 * 6
* 單一小區最大使用者數 : 4, 6, 8
* 使用者與基站距離 : r ∈ [0.01, 1] km
* 基地站干擾範圍 : 相隔兩區內的蜂巢基站都會互相干擾
* 人數模擬 : Poisson 分布 (𝜆 = 2)

## 訊號干擾雜訊比(SINR)
* 目標 : 讓蜂巢基站的整理 SINR 最大化
![image](https://github.com/jamieeeeeeee/Power-Allocation-/blob/main/SINR.png)

## 模型
### DQN
強化學習是機器學習(Machine learning)的一種，指的是電腦透過與一個動態(dynamic)環境不斷重複地互動，來學習正確地執行一項任務。這種嘗試錯誤(trial-and-error)的學習方法，使電腦在沒有人類干預、沒有被寫入明確的執行任務程式下，就能夠做出一系列的決策。
![image](https://github.com/jamieeeeeeee/Power-Allocation-/blob/main/DQN.png)

* Agent部份(大腦) :
    * 會將environment環境每一個時間點的observation(觀察)的集合當作環境的狀態(State)
    * 從環境的狀態(State)跟reward(獎勵)再去選擇一個最好的action(動作)，稱為policy(策略)
* Environment部份(環境) :
    * 會接收Agent執行的action(動作)，並吐出reward跟observation給agent。


## 模型效能比較(四種算法)
* Fractional Programming
* Weighted MMSE
* Max power
* Random power


## 結果
![image](https://github.com/jamieeeeeeee/Power-Allocation-/blob/main/result.png)