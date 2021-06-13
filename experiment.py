import os,sys, argparse
from time import time
from bisect import bisect_left

import json

import numpy as np
import pandas as pd
# import matplotlib.pyplot as plt
# from seaborn import heatmap
try:
    from tqdm import tqdm
    succeed_importing_tqdm = True 
except:
    succeed_importing_tqdm = False

from fpylll import *


'''
usage:
python3 experiment.py --type_of_problem LWR --max_n 20 --min_n 20 --max_alpha 0.03 --min_alpha 0.03 --max_iter 100 --method Bai_Galbraith --BKZ_beta 30 --m_times 2

type_of_problem : ‘LWE’か‘LWR’か
max_n , min_n : 実験を行う次元n.  max_nとmin_nが異なる時にはmin_n≦n≦max_n なるnを5刻みで実行
max_alpha ,min_alpha : 実験を行うα.  異なる時は0.003刻みで実行
max_iter : 繰り返し回数
method : 埋め込み方法 ‘Kannan’か‘Bai_Galbraith’か
BKZ_beta : BKZアルゴリズムのパラメータβ

--m_times 2
のようにすると m = 2n みたいにできる

'''

def rounding(a):
    x = np.round(a)+(((a%1)==0.5)*(a//1%2==0)).astype('int')
    return x


## エラトステネスの篩
def sieve(n):
    flag_array = np.ones(n+1,dtype='bool')
    flag_array[:2] = 0,0
    for i in range(2,int(n**0.5)+2):
        if not flag_array[i]:
            continue
        counter = 2*i
        while counter<=n:
            flag_array[counter] = 0
            counter += i
    array=[]
    for i in range(n+1):
        if flag_array[i]:
            array.append(i)
    return array

## nより大きい最小の素数
def next_prime(n):
    primes = sieve(int(1.1*n)+3)
    return primes[bisect_left(primes,n+1)]

## 丸め正規分布
def round_gauss(mean=0,std=1,size=1):
    sampled = np.random.normal(loc=mean,scale=std,size=size)
    return rounding(sampled)

def center(a,q):
    half_q = q/2
    half_q = round(half_q) +(1 if (half_q)//1==0.5 and (half_q)%2==0 else 0) 
    return (a%q+half_q)%q - half_q

## 正規分布
def gaussian(x,sigma):
    return np.exp(-(x**2)/(2*sigma**2))

## LWRの鍵生成
def LWR_create_key(m,n,q,p,sigma):    
    A = np.random.randint(q,size=(m,n))
    s = round_gauss(0,sigma,size=n).reshape((-1,1))%q
    b = rounding(p/q*(A.dot(s)%q))
    return A,b,s

## LWE
def LWE_create_key(m,n,q,sigma,sigma_s=None):    
    A = np.random.randint(q,size=(m,n))
    if sigma_s is None:
        s = np.random.randint(q,size=(n,1))%q
    else:
        s = round_gauss(0,sigma_s,size=n).reshape((-1,1))%q
    e = round_gauss(0,sigma,size=m).reshape((-1,1))
    b = ( A.dot(s) + e )%q
    return A,b,s,e

## BKZ
def BKZ_from_matrix(B:np.array,bkz_param=3) -> np.array : 
    try:
        if B.dtype != np.dtype('int64'):
            B = B.astype(np.int)
        B_new = IntegerMatrix(B.shape[0],B.shape[1])
        B_new.set_matrix(B.tolist())
        B_bkz = BKZ.reduction(B_new, BKZ.Param(bkz_param))
        successfully_done = True
    except BaseException as exception:
        successfully_done = False
        return None,successfully_done,exception
    return B_bkz.to_matrix(np.empty_like(B)), successfully_done, None

## 篩法
def Gauss_sieve_from_matrix(B:np.array) -> np.array : 
    try:
        if B.dtype != np.dtype('int64'):
            B = B.astype(np.int)
        B_new = IntegerMatrix(B.shape[0],B.shape[1])
        B_new.set_matrix(B.tolist())
        shortest = GaussSieve(B_new, algorithm=2,verbose=True)()
        B_sieve = B_new.to_matrix(np.empty_like(B))
#         B_sieve = B_sieve[(B_sieve**2).sum(axis=1).argsort()].copy()
        successfully_done = True
    except BaseException as exception:
        successfully_done = False
        return None,successfully_done,exception,None
    return B_sieve , successfully_done, None,shortest

def euclid(x,y,p):# p is the order of F_p, x > y
    if x <= y:
        swap_flag = True
        x,y = y,x
    else:
        swap_flag = False
    if y==0:
        return x,1,0
    q,r = x//y, x%y
    gcd,u,v = euclid(y,r,p)
    if swap_flag:
        return gcd,(u-q*v)%p,v
    return gcd,v,(u-q*v)%p

def inv_in_finite_field(a,p):
    if a >= p or a<0:
        a = a%p
    a = rounding(a)
    gcd,a_inv,_ = euclid(a,p,p)
    return a_inv

def Gauss_elimination(A,p):
#     assert A.shape[0]==A.shape[1], f'expected square matrix: {A.shape}'
    m,n = A.shape
    A = A.copy()
    for i in range(min(n,m)): #col
        a_ii_inv = inv_in_finite_field(A[i,i],p)
        A[i,:] = (a_ii_inv*A[i,:])%p
        for j in range(m): #row
            if i==j: continue
            q = A[j,i]
            A[j,:] = ((A[j,:]-q*A[i,:])%p)
    return A

# inverse of matrix A in finite field p
def inv_matrix_in_finite_field_by_Gauss_elimination(A,p):
    assert A.shape[0]==A.shape[1], f'expected square matrix: {A.shape}'
    n = A.shape[0]
    A = A.copy()
    A_inv = np.eye(n)
    for i in range(n): #col
        a_ii_inv = inv_in_finite_field(A[i,i],p)
        A[i,:] = (a_ii_inv*A[i,:])%p
        A_inv[i,:] = (a_ii_inv*A_inv[i,:])%p
        for j in range(n): #row
            if i==j: continue
            q = A[j,i]
            A[j,:] = ((A[j,:]-q*A[i,:])%p)
            A_inv[j,:] = ((A_inv[j,:]-q*A_inv[i,:])%p)
    return A_inv

def compute_p(q,sigma):
    numerator = q + q*np.sqrt(12*sigma**2+1)
    denominator = 12*sigma**2
    return int(numerator/denominator)+1

def compute_sigma_from_p(q,p):
    sigma = np.sqrt((q**2+2*p*q)/(12*p**2))
    return sigma

# def compute_sigma_from_p(q,p): ## 
#     sigma = np.sqrt((q**2-p**2)/(12*p**2))
#     return sigma

# def compute_p(q,sigma):
#     p = q/np.sqrt(12*(sigma**2)+1)
#     return p

def attack_Kannan(A,b,q,M=1,BKZ_beta=3):
    A,b = A.copy(),b.copy()
    m,n = A.shape
    ## 行列構成
    A_2prime = Gauss_elimination(A.T,q)
    B = np.vstack((A_2prime,np.hstack([np.zeros((m-n,n)),np.eye(m-n)*q]))).astype(int)
    B_prime = np.hstack([B,np.zeros((m,1))])
    B_prime = np.vstack([B_prime,np.hstack([b.T,[[M]]])]).astype(np.int)
    
    ## BKZ
    reduced_B_prime,successfully_done,exception = BKZ_from_matrix(B_prime,bkz_param=BKZ_beta)
    if successfully_done:
        e = reduced_B_prime[0,:-1].reshape((-1,1))
        As = (b-e)%q
        s = (inv_matrix_in_finite_field_by_Gauss_elimination(A[:n,:n],q).dot(As[:n]))%q
        return s,e,successfully_done,exception
    else:
        return None,None, successfully_done,exception

def attack_Bai_Galbraith(A,b,q,nu,M=1,BKZ_beta=3):
    A,b = A.copy(),b.copy()
    m,n = A.shape
    ## 行列構成
    B = np.vstack([np.hstack([nu*np.eye(n),    A.T,         np.zeros((n,1))]),
                   np.hstack([np.zeros((m,n)), q*np.eye(m), np.zeros((m,1))]),
                   np.hstack([np.zeros((1,n)), b.T,         [[M]]])]).astype(int)
    reduced_B_prime,successfully_done,exception = BKZ_from_matrix(B,bkz_param=BKZ_beta)

    if successfully_done:
        # row_number = 0
        if np.abs(reduced_B_prime[0,:n]).sum() != 0:
            row_number = 0
            print()
        else:
            row_number = 1

        e = reduced_B_prime[row_number,n:-1].reshape((-1,1))
        s = reduced_B_prime[row_number,:n].reshape((-1,1))
        As = (b-e)%q
        second_shortest = reduced_B_prime[row_number+1]
        return s,e,successfully_done,exception,reduced_B_prime
    else:
        return None,None, successfully_done,exception,None



def make_all_dir_in_path(path):
    dirs = path.split('/')
    d = dirs[0]
    for i in range(1,len(dirs)):
        d = ''.join([d,'/',dirs[i]])
        if not os.path.exists(d):
            os.mkdir(d)

def challenge(problem_type, max_n,max_alpha,max_iter,method,BKZ_beta=30, m_times=2,embedding_M=1,min_alpha=None,min_n=None,working_directory="./"):
    assert method in ['Kannan','Bai_Galbraith','Sieve'], f"method should be in ['Kannan','Bai_Galbraith','Sieve']] : {method}"
    assert problem_type in ['LWE','LWR'], f"problem_type should be in ['LWE','LWR'] : {problem_type}"
    
    if min_alpha is None:
        min_alpha = 0.001
    if min_n is None:
        min_n = 5
    n_step_size = 5
    alpha_step_size = 5

    if m_times % 1 == 0:
        m_times = int(m_times)
    
    path = os.path.join(working_directory,problem_type,'results',method)
    

    reduction_method = 'BKZ'
    columns = ['sigma','done_iter_num','total_time','total_success','success_gap','success_gap_count','fail_gap','fail_gap_count' ]

    ## 保存用のパラメータ設定
    if isinstance(BKZ_beta,str):
        beta_dynamic = True
        beta_divisor = int(BKZ_beta.split('/')[-1])
        param = f'beta=n_div_{beta_divisor}'
    else:
        beta_dynamic = False
        param = f'beta={BKZ_beta}'

    if method == 'Sieve':
        param = ''

    if isinstance(m_times,int) or isinstance(m_times,float):
        param = '_'.join([param,f'm={m_times}n']) if param != '' else f'm={m_times}n'
    else:
        param = '_'.join([param,f'm=n^2'])

    # make_all_dir_in_path(os.path.join(path,param))


    # counter1 = tqdm(range(min_n,max_n+1,n_step_size)) if succeed_importing_tqdm else range(min_n,max_n+1,n_step_size)
    counter1 = range(min_n,max_n+1,n_step_size)
    ## 各次元nについて繰り返し
    for n in counter1:            
        ## パラメータ決定
        q = next_prime(n**2)
        
        if m_times:
            m = int(n*m_times)
        else:
            m = n**2
        if beta_dynamic:
            BKZ_beta = max(int(np.round(n/beta_divisor)),5)
        
        csv_path = os.path.join(path,f'{param}/n_{n}_m_{m}.csv')
        
        ## 過去のデータがあれば読み込み, なければ空DFを作る
        if os.path.exists(csv_path):
            df = pd.read_csv(csv_path,index_col=0)
        else:
            df = pd.DataFrame([],columns=columns,index=[])
        values = df.values
        index = df.index.tolist()
        

        try:
            # counter2 = tqdm(range(int(min_alpha*1000),int(max_alpha*1000)+1,3)) if succeed_importing_tqdm else range(int(min_alpha*1000),int(max_alpha*1000)+1,3)
            counter2 = range(int(min_alpha*1000),int(max_alpha*1000)+1,alpha_step_size)
            for i in counter2:
                #パラメータ決定
                alpha = np.round(i/1000,4)
                sigma = np.round(q*alpha,6)
                if problem_type=='LWR':
                    p = int(compute_p(q,sigma))
                
                #過去の履歴のロード
                alpha_already_done_flag = False
                for j in df.index:
                    if abs(alpha-j)<0.0000001:
                        alpha_already_done_flag = True
                        alpha = j
                        break
                if alpha_already_done_flag:
                    alpha_idx = df.index.get_loc(alpha)
                    done_iter_num = int(values[alpha_idx,1])
                else:
                    done_iter_num = 0
                    index = index + [alpha]
                    alpha_idx = values.shape[0]
                    values = np.vstack([values,[sigma,0,0,0,0,0,0,0]])

                #実験
                if problem_type=='LWE':
                    print(f'n={n:>2}, m={m:>3}, alpha:{alpha:>.4}, q:{q:>4}')
                else:
                    print(f'n={n:>2}, m={m:>3}, alpha:{alpha:>.4}, q:{q:>4}, p:{p:>4}')
                counter3 = tqdm(range(max_iter-done_iter_num)) if succeed_importing_tqdm else range(max_iter-done_iter_num)
                for _ in counter3:
                    # np.random.seed(1234)
                    ## 問題作成
                    if problem_type=='LWE':
                        A,b,s,e = LWE_create_key(m,n,q,sigma,sigma)
                    else:
                        A,c,s = LWR_create_key(m,n,q,p,sigma)
                        b = rounding(q/p*c.copy().astype('float'))
                        e = center((b - A.dot(s)),q).flatten()
                    
                    
                    
                    
                    ## 攻撃
                    if method == 'Kannan':
                        BKZ_beta = min(BKZ_beta, m+1)
                        start = time()
                        s_,e_,successfully_done,exception = attack_Kannan(A,b,q,embedding_M,BKZ_beta)
                        if not successfully_done:
                            raise exception
                        elapsed_time = time() - start
                    elif method == 'Bai_Galbraith':
                        BKZ_beta = min(BKZ_beta, m+n+1)
                        start = time()
                        nu = 1 #if problem_type=='LWE' else round(compute_sigma_from_p(q,p)/sigma)
                        s_,e_,successfully_done,exception,B_reduced = attack_Bai_Galbraith(A,b,q,nu=1,M = embedding_M,BKZ_beta=BKZ_beta)
                        if not successfully_done:
                            raise exception
                        elapsed_time = time() - start
                    elif method == 'Sieve':
                        start = time()
                        nu = 1
                        s_,e_,successfully_done,exception = attack_Bai_Galbraith_sieve(A,b,q,nu)
                        if not successfully_done:
                            raise exception
                        elapsed_time = time() - start
                    else:
                        raise Exception(f'未実装でしょ! : {method}')
                
                    
                    params = {'n':n, 'm':m, 'q':q, 'p':p, 'sigma':sigma}
                    dic = {'params':params, 'A':A.tolist(), 'b':b.flatten().tolist(), "s":s.flatten().tolist(), "B":B_reduced.tolist()}
                    
                    if not os.path.exists('./result_/'):
                        os.mkdir('./result_/')
                    folder = f'./result_/n_{n}_m_{m}_q_{q}_p_{p}/'
                    if not os.path.exists(folder):
                        os.mkdir(folder) 
                    num = [i for i in os.listdir(folder) if '.json' in i]
                    num = 0 if num==[] else max(list(map(lambda x:int(x.split('.')[0]),num)))
                    with open(os.path.join(folder,f'{num+1}.json'),'w') as f:
                        json.dump(dic,f)



                        # fail_case_path = csv_path.replace('results','fail_case')
                        # fail_case_path = '/'.join(fail_case_path.split('/')[:-1]+[f'/alpha={alpha}'])
                        # if not os.path.exists(fail_case_path):
                        #     make_all_dir_in_path(fail_case_path)
                        # with open(fail_case_path+f'/n_{n}_m_{m}.csv','a') as f:
                        #     # key = np.hstack([s,None,e,None,[1]]).astype('int')
                        #     # obtained =  np.hstack([s_,None,e_,None,[1]]).astype('int')
                        #     # f.write('key,'+','.join(key.astype('str'))+'\n')
                        #     # f.write('obtained,'+','.join(obtained.astype('str'))+'\n')

                        #     f.write('key,'+','.join(s.astype('int').astype('str'))+',,'+','.join(e.astype('int').astype('str'))+',,'+'1\n')
                        #     f.write('obtained,'+','.join(s_.astype('str'))+',,'+','.join(e_.astype('str'))+',,'+'1\n')
                        #     f.write('gap,'+str(gap)+'\n\n')
                        #     f.flush()




                    # print('b',b.flatten())
                    # print(b.shape,A.shape,s.shape,e.shape)
                    # print((b-A.dot(s)-e.reshape((m,1))).flatten()%q)
                    # print('\n\n\n\n')
                    s,s_ = center(s,q),center(s_,q)
                    # e,e_ = center(e,q),center(e_,q)
                    # print('e',e.flatten(),e.mean(),e.std())
                    # print('s',center(s.flatten(),q))
                    # print('\n\n\n')
                    # print('e_',e_.flatten())
                    # print('s_',s_.flatten())
                    # print('e_diff',center((e_-e).flatten()%q,q))
                    # print('e_diff',center((e_+e).flatten()%q,q))
                    # print('e_diff',(e_.flatten()-e.flatten())%q)
                    # print('e_diff',(e_.flatten()+e.flatten())%q)
                    # print('|e|^2_diff', ((e.T.dot(e)+s.T.dot(s))-(e_.T.dot(e_)+s_.T.dot(s_)))[0,0]/(n+m))
                    # print('norm',(e_.T.dot(e_)+s_.T.dot(s_)+1))
                    # print(e.flatten(),'std',e.flatten().std())
                    # print('correct norm',(e.T.dot(e)+s.T.dot(s)+1))
                    # print('\n\n\n')
                    
        # Keyboard Interruption をした時に保存する用
        except BaseException as e:
            # df_after_done = pd.DataFrame(values,index=index,columns=columns)
            # df_after_done.index.name = 'alpha'
            # df_after_done = df_after_done.sort_index()
            # df_after_done.to_csv(csv_path)
            raise e
        
        # df_after_done = pd.DataFrame(values,index=index,columns=columns)
        # df_after_done.index.name = 'alpha'
        # df_after_done = df_after_done.sort_index()
        # df_after_done.to_csv(csv_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--type_of_problem', '-t', type=str, help="type: Choose from ['LWE','LWR']",
                        choices=['LWE','LWR'],required=True)
    parser.add_argument('--max_n', '-N', type=int, help='maximum n (dimension) to make problems',
                        required=True)
    parser.add_argument('--min_n', '-n', type=int, help='minimum n (dimension) to make problems',
                        required=False, default=20)
    parser.add_argument('--max_alpha', '-A', type=float, help='maximum alpha (sigma/q) to make problems',
                        required=True)
    parser.add_argument('--min_alpha', '-a', type=float, help='minimum alpha (sigma/q) to make problems',
                        required=False, default=0.01)
    parser.add_argument('--max_iter', '-i', type=int, help='number of iteration',
                        required=True)
    parser.add_argument('--method', '-mth', type=str, help='method',
                        required=True, choices=['Kannan','Bai_Galbraith','Sieve'])
    parser.add_argument('--working_directory', '-w', type=str, help='working_directory',
                        required=False, default='./')
    parser.add_argument('--BKZ_beta', '-b', type=int, help='parameter beta of BKZ',
                        required=True)
    parser.add_argument('--m_times', '-m', type=float, help='ratio of n and m (m/n)',
                        required=False, default=None)
    args = parser.parse_args()

    if args.m_times is not None:
        challenge(problem_type = args.type_of_problem,
                max_n = args.max_n, max_alpha = args.max_alpha, max_iter = args.max_iter,
                method = args.method ,BKZ_beta=args.BKZ_beta, m_times = args.m_times, embedding_M=1,
                min_alpha = args.min_alpha, min_n = args.min_n, working_directory = args.working_directory)
    else:
        for m_times in [1.8,1.85,1.9,1.95,2,2.25,2.5,2.75,3]:
            challenge(problem_type = args.type_of_problem,
                max_n = args.max_n, max_alpha = args.max_alpha, max_iter = args.max_iter,
                method = args.method ,BKZ_beta=args.BKZ_beta, m_times = m_times, embedding_M=1,
                min_alpha = args.min_alpha, min_n = args.min_n, working_directory = args.working_directory)