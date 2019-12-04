import operator

class Node:
    def __init__(self,val=[]):
        self.val = val
        self.children = {}

class HashTree:
    def __init__(self, prime=3, valNum=3):
        self.root = Node([])    # 建立根节点
        self.prime = prime
        self.valNum = valNum
        
    def _hash_fun(self,x):
        return x % self.prime
        
    def insert(self,val):    # 给定一个值val,创建一个值为val的节点并将其插入到树中
        root = self.root
        i = 0
        while root.children:
            # if root has children
            if self._hash_fun(val[i]) in root.children.keys():    # 如果想找的孩子存在
                root = root.children[self._hash_fun(val[i])]
                i += 1
             
            # if the child doesn't exist
            else:
                root.children[self._hash_fun(val[i])] = Node([val])
                return
        
        # so the root doesn't have child
        # the value of root is smaller than self.prime
        if len(root.val) < self.prime:
            root.val.append(val)
            return
        
        #root的值的个数已经等于self.prime了，那么我们必须split root
        else:
            #先把val加入到root的值当中
            if i >= self.prime:
                print("hash crashed!")
                print("Old Val: ",root.val)
                print("New Val: ",val)
                return
            root.val.append(val)
            
            #对于root值中的的每一项
            for item in root.val:
                # 表示余数是多少
                j = self._hash_fun(item[i])
                if j in root.children.keys():    #如果余数已经在其中了
                    root.children[j].val.append(item)    #将item加入到列表中
                else:
                    root.children[j] = Node()    #首先创建该节点
                    root.children[j].val = [item]    #如果没有，则创建新列表
                root.val = []
 
                if len(root.children[j].val) > self.prime:    #分裂节点后，仍然出现了大于3的情况，那么我们需要继续分裂节点
                    i += 1
                    root = root.children[j]
                    for item in root.val:    #对于root值中的的每一项
                        j = self._hash_fun(item[i])
#                         j = (item[i]-1) % 3    #表示余数是多少
                        if j in root.children.keys():    #如果余数已经在其中了                            
                            root.children[j].val.append(item)    #将item加入到列表中
                        else: 
                            root.children[j] = Node()
                            root.children[j].val = [item]    #如果没有，则创建新列表
                    root.val = []
                    
    def isExists(self, item):
        root = self.root
        i=0
        while root.children:
            j = self._hash_fun(item[i])
            if j in root.children.keys():
                root = root.children[j]
                i+=1
            else:
                return False
            
        for v in root.val:
            if operator.eq(item,v):
                return True
        return False
   
    def _hash_item(self, item):
        item=["%02d"%i for i in item]
        return ''.join(item)
        
    def isContained(self, sequence):
        ContainedItem=[]
        hash_t = []
        for n in range(len(sequence)-self.valNum+1):
            seq_seg = list(sequence[n:n+self.valNum])
            if self.isExists(seq_seg):
                hash_seq = self._hash_item(seq_seg)
                if hash_seq not in hash_t:
                    hash_t.append(hash_seq)
                    ContainedItem.append(seq_seg)
                       
        return ContainedItem
    
    def PrintTree(self,node):    #如何层次的输出树
        if not node.val:    #如果这个节点的值不存在
            if not node.children:
                return    #如果孩子也没有
            else:    #有孩子了
                res = []
                for item in node.children.values():
                    res += [self.PrintTree(item)]
        #这个节点的值存在
        else:
            return node.val
        return res