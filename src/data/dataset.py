import numpy as np
import random
import torch
from joblib import Parallel, delayed
from scipy.stats import rankdata
from torch.utils.data import Dataset, Sampler



class MyCollator(object):
    def __init__(self, mysql_url):
        self.mysql_url = mysql_url
        
    def __call__(self, examples):
        rows = examples
        ase_crystal_list = [row.toatoms() for row in rows]
        nodes = [item.get_atomic_numbers() for item in ase_crystal_list]
        wf = np.array([np.array(item.data['work_func'][:, 1]) for item in rows])
        
        max_len = np.max([len(item) for item in nodes])

        # Compute geometric features
        distance_matrix = [item.get_all_distances(mic=True) for item in ase_crystal_list]
        node_extra = [setGraphFea(ase_crystal_list[i], distance_matrix[i]) for i in range(len(rows))]
        distance_matrix_trimmed = [mask_elements_over_threshold(item, 8.0) for item in distance_matrix]
        distance_embd = [GBF_distance_encode(item, 0.0, 8.0, 12) for item in distance_matrix_trimmed]

        # Compute padding parameters
        batch_size = len(nodes)

        # Initialize padded containers
        nodes_padded = np.zeros([batch_size, max_len])
        distance_padded = np.zeros([batch_size, max_len, max_len, 12])
        nodes_extra_padded = np.zeros([batch_size, max_len, 3 * 11 * 11])

        # Fill padded data
        for i, item in enumerate(nodes):
            len_temp = len(item)
            nodes_padded[i, : len_temp] = item
            distance_padded[i, :len_temp, :len_temp, :] = distance_embd[i]
            nodes_extra_padded[i, :len_temp, :] = node_extra[i]

        return {
            'nodes': torch.tensor(nodes_padded).long(),
            'wf': torch.tensor(wf).float(),
            'distance': torch.tensor(distance_padded).float(),
            'node_extra': torch.tensor(nodes_extra_padded).float(),
        }




def setGraphFea(atom, distance):

    neighbors = 11
    num_atoms = len(distance)
    if num_atoms == 1:
        return np.zeros((num_atoms, 3 * neighbors * neighbors))
    embedding = np.zeros((num_atoms, 3 * neighbors * neighbors))
    if len(distance) < neighbors + 1:
        neighbors = num_atoms - 1
    sorted_idx = np.argsort(distance, axis=1)
    idx_cut = sorted_idx[:, 1:neighbors+1]

    i_indices = np.repeat(np.arange(num_atoms), neighbors**2)
    j_indices = np.repeat(np.expand_dims(idx_cut, axis=2), neighbors, axis=2).flatten()
    k_indices = idx_cut[idx_cut].flatten()

    angle_indices = [(i, j, k) for i, j, k in zip(i_indices, j_indices, k_indices)]
    # Vectorized angle calculation (assuming get_angles can accept arrays of tuples)
    angles = atom.get_angles(angle_indices)
    cosines = np.cos(np.radians(angles))

    angle_embedding = np.array(cosines).reshape(num_atoms, neighbors * neighbors)
    edge_ij = distance[i_indices, j_indices].reshape(num_atoms, neighbors * neighbors)
    edge_jk = distance[j_indices, k_indices].reshape(num_atoms, neighbors * neighbors)


    # Populate the output array
    for n in range(neighbors**2):
        embedding[:, 3*n] = edge_ij[:, n]
        embedding[:, 3*n + 1] = edge_jk[:, n]
        embedding[:, 3*n + 2] = angle_embedding[:, n]

    return embedding





def GBF_distance_encode(matrix, min, max, step):

    gamma = (max - min) / (step - 1)
    filters = np.linspace(min, max, step)
    matrix = matrix[:, :, np.newaxis]
    matrix = np.tile(matrix, (1, 1, step))
    matrix = np.exp(-((matrix - filters) ** 2) / gamma**2)
    
    return matrix




def mask_elements_over_threshold(matrix, threshold):
    """Replace elements exceeding threshold with 0."""
    return np.where(matrix > threshold, 0.0, matrix)




def threshold_sort(matrix, threshold, neighbors, reverse=False, adj=False):
    mask = matrix > threshold
    distance_matrix_trimmed = np.ma.array(matrix, mask=mask)

    if reverse:
        distance_matrix_trimmed = rankdata(
            distance_matrix_trimmed * -1, method="ordinal", axis=1
        )
    else:
        distance_matrix_trimmed = rankdata(
            distance_matrix_trimmed, method="ordinal", axis=1
        )
    distance_matrix_trimmed = np.nan_to_num(
        np.where(mask, np.nan, distance_matrix_trimmed)
    )
    distance_matrix_trimmed[distance_matrix_trimmed > neighbors + 1] = 0

    if not adj:
        distance_matrix_trimmed = np.where(
            distance_matrix_trimmed == 0, distance_matrix_trimmed, matrix
        )
        return distance_matrix_trimmed

    # adj == True case
    adj_list = np.zeros((matrix.shape[0], neighbors + 1))
    adj_attr = np.zeros((matrix.shape[0], neighbors + 1))
    for i in range(0, matrix.shape[0]):
        temp = np.where(distance_matrix_trimmed[i] != 0)[0]
        adj_list[i, :] = np.pad(
            temp,
            pad_width=(0, neighbors + 1 - len(temp)),
            mode="constant",
            constant_values=0,
        )
        adj_attr[i, :] = matrix[i, adj_list[i, :].astype(int)]
    distance_matrix_trimmed = np.where(
        distance_matrix_trimmed == 0, distance_matrix_trimmed, matrix
    )
    return distance_matrix_trimmed, adj_list, adj_attr






def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def metic_score(datas=np.array([ 5.9138,  0.    ,  1.1654,  0.    ,  7.3021,  0.    ,  1.1654, 0.    , 10.921 ])):
    """
    input datas: 1D array -- 9 elements -- this version
    only totl elec accepeted
    """
    eigenvalues, eigenvectors = np.linalg.eig(datas.reshape(3,3))
    D = np.diag(eigenvalues)
    return np.mean(np.diag(D))



def schmidt_orthogonalization(vectors):
    vectors = vectors.reshape(3, 3)
    eigenvalues, eigenvectors = np.linalg.eig(vectors)
    diagonal_matrix = np.real(np.diag(eigenvalues))
    ele = np.mean(np.real(eigenvalues))

    return ele




@torch.no_grad()
def sample_from_model(model, x, points=None, test=False):
    """Sample from model and compute predictions."""
    device = 'cpu'
    x = x.to(device)
    points = points.to(device)

    model = model.to(device)
    model.eval()
    N = len(x)
    _, predicts, _ = model(points, points=x)
    predicts = predicts.cpu()
    topN = 5
    targets = torch.tensor(np.arange(N))
    _, indx = predicts.topk(topN, 0)
    indx = indx.t()
    indx = indx.cpu().numpy().tolist()

    vals = []
    for idxs in indx:
        val = [schmidt_orthogonalization(x[i, :].cpu().numpy()) for i in idxs]
        vals.append(val)
    vals = np.array(vals)
    print(vals)
    if test:
        accuracy_sum = 0
        targets = targets.cpu().numpy()
        judge = torch.nn.MSELoss(reduce=True, size_average=True)
        error_diag = []
        error_diag_mae = []
        ground_truth = []
        best_pred = []
        for i in range(N):
            top3_captions = [indx[i][j] for j in range(len(indx[i]))]
            err = [judge(x[i, :], x[j, :]).cpu().numpy() for j in top3_captions]
            err_diag = [(schmidt_orthogonalization(x[i, :].cpu().numpy()) - schmidt_orthogonalization(x[j, :].cpu().numpy())) ** 2 for j in top3_captions]
            err_diag_mae = [
                abs(schmidt_orthogonalization(x[i, :].cpu().numpy()) - schmidt_orthogonalization(x[j, :].cpu().numpy()))
                for j in top3_captions]
            error_diag.append(np.min(err_diag))
            error_diag_mae.append(np.min(err_diag_mae))
            ground_truth.append(schmidt_orthogonalization(x[i, :].cpu().numpy()))
            best_pred.append(schmidt_orthogonalization(x[top3_captions[np.argmin(err_diag)], :].cpu().numpy()))
            if np.min(err) <= 1.5:
                accuracy_sum += 1

        best_pred = np.array(best_pred)
        ground_truth = np.array(ground_truth)
        rss = np.sum((ground_truth - best_pred) ** 2)

        # Compute total sum of squares
        tss = np.sum((ground_truth - np.mean(ground_truth)) ** 2)

        # Compute R2 score
        r2 = 1 - (rss / tss)

        import matplotlib.pyplot as plt

        plt.scatter(ground_truth, best_pred, s=10)
        plt.plot([min(ground_truth), max(ground_truth)],
                [min(ground_truth), max(ground_truth)], 'r--', label='y=x')
        plt.xlabel('True Values')
        plt.ylabel('Predictions')
        plt.title('Scatter Plot of Predicted Values')
        plt.legend()
        plt.savefig('test_set.svg', format='svg')
        plt.show()
        return r2



class CharDataset(Dataset):
    def __init__(self, rows):
        self.rows = rows
        self.data_size = len(rows)
        print(f'data has {self.data_size} examples')

    def __len__(self):
        return self.data_size

    def __getitem__(self, idx):
        return self.rows[idx]
    

def processDataFiles(files):
    text = ""
    for f in tqdm(files):
        with open(f, 'r') as h:
            lines = h.read() # don't worry we won't run out of file handles
            if lines[-1]==-1:
                lines = lines[:-1]
            #text += lines #json.loads(line)
            text = ''.join([lines,text])
    return text



def get_all_atoms(lst, mysql_url, mp_n):
    k, m = divmod(len(lst), mp_n)
    split_idx = [lst[i*k+min(i, m):(i+1)*k+min(i+1, m)] for i in range(mp_n)]
    results = Parallel(n_jobs=mp_n)(delayed(query_database)(mysql_url, idx_list) for idx_list in split_idx)
    rows = [item for sublist in results for item in sublist]
    rows = [row.toatoms() for row in rows]
    return rows





class SynchronizedSampler(Sampler):
    def __init__(self, data_source):
        self.data_source = data_source

    def __iter__(self):
        indices = list(range(len(self.data_source)))
        torch.random.manual_seed(42)
        indices = torch.randperm(len(indices)).tolist()
        return iter(indices)

    def __len__(self):
        return len(self.data_source)


