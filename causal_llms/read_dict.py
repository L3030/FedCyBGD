import pickle

if __name__ == '__main__':
    random_path = f'./outputs/llama2-7b/64_3_random/outdict.pkl'
    fixed_path = f'./outputs/llama2-7b/64_3_fixed/outdict.pkl'

    with open(random_path, 'rb') as f:
        randomdic = pickle.load(f)
    with open(fixed_path, 'rb') as f:
        fixeddic = pickle.load(f)
    print('--------------fixed_block_client_eval_loss-----------------')
    print(fixeddic)
    print('--------------random_block_client_eval_loss-----------------')
    print(randomdic)