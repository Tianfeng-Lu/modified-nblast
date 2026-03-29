import numpy as np
import trimesh
import matplotlib.pyplot as plt
from pycpd import RigidRegistration

class GroupRigidRegistration(RigidRegistration):
    def __init__(self, target_group_indicator=None, source_group_indicator=None, translate=True, scale=False, *args, **kwargs):
        '''
        target_group_indicator: list of integers, indicating the length of each group in the target point cloud
        source_group_indicator: list of integers, indicating the length of each group in the source point cloud
        '''
        super().__init__(*args, **kwargs)
        if target_group_indicator is None or source_group_indicator is None:
            target_group_indicator = [len(self.X)]
            source_group_indicator = [len(self.Y)]
        self.target_group_indicator = target_group_indicator
        self.source_group_indicator = source_group_indicator
        self.translate = translate
        self.scale = scale
        if len(self.target_group_indicator) != len(self.source_group_indicator):
            raise ValueError(f"The # target group {len(self.target_group_indicator)} should be equal to # source group {len(self.source_group_indicator)}")
        if len(self.X) != sum(self.target_group_indicator):
            raise ValueError(f"The sum of target_group_indicator {len(self.X)} should be equal to the length of the target point cloud {sum(self.target_group_indicator)}")
        if len(self.Y) != sum(self.source_group_indicator):
            raise ValueError(f"The sum of source_group_indicator {len(self.Y)} should be equal to the length of the source point cloud {sum(self.source_group_indicator)}")
        # generate the mask for each group
        self.group_mask = np.zeros((sum(target_group_indicator), sum(source_group_indicator)))
        arr1 = np.insert(np.cumsum(target_group_indicator), 0, 0)
        arr2 = np.insert(np.cumsum(source_group_indicator), 0, 0)
        for i, j, k, l in zip(arr1[0:-1], arr1[1:], arr2[0:-1], arr2[1:]):
            self.group_mask[np.ix_(np.arange(i, j), np.arange(k, l))] = 1
        self.group_mask = self.group_mask.T # (M, N) = (#source, #target)
        self.Ms = np.array(target_group_indicator)
        self.Ns = np.array(source_group_indicator)

    def expectation(self):
        P = np.sum((self.X[None, :, :] - self.TY[:, None, :]) ** 2, axis=2)

        c = (2 * np.pi * self.sigma2) ** (self.D / 2)
        c = c * self.w / (1 - self.w)
        c = c * self.Ms / self.Ns # change this to group-wise
        gc = np.zeros((1, P.shape[1]), dtype=float)
        arr1 = np.insert(np.cumsum(self.target_group_indicator), 0, 0)
        for i, c_value in enumerate(c):
            start = arr1[i]
            end = arr1[i+1]
            gc[:, start:end] = c_value

        P = np.exp(-P / (2 * self.sigma2))
        P = P * self.group_mask # mask all irrelevant point-point connections
        den = np.sum(P, axis = 0, keepdims = True) # (1, N)
        den = np.clip(den, np.finfo(self.X.dtype).eps, None) + gc # add the group-wise constant

        self.P = np.divide(P, den)
        self.Pt1 = np.sum(self.P, axis=0)
        self.P1 = np.sum(self.P, axis=1)
        self.Np = np.sum(self.P1)
        self.PX = np.matmul(self.P, self.X) # updated 0117: forgot to add it ... 

    def update_transform(self):
        """
        Calculate a new estimate of the rigid transformation.

        """
        if self.translate:
            # target point cloud mean
            muX = np.divide(np.sum(self.PX, axis=0), self.Np) # updated 0117: use PX instead of computing it again
            # source point cloud mean
            muY = np.divide(
                np.sum(np.dot(np.transpose(self.P), self.Y), axis=0), self.Np)

            self.X_hat = self.X - np.tile(muX, (self.N, 1))
            # centered source point cloud
            Y_hat = self.Y - np.tile(muY, (self.M, 1))
        else: # no translation
            self.X_hat = self.X
            Y_hat = self.Y

        self.YPY = np.dot(np.transpose(self.P1), np.sum(
            np.multiply(Y_hat, Y_hat), axis=1))

        self.A = np.dot(np.transpose(self.X_hat), np.transpose(self.P))
        self.A = np.dot(self.A, Y_hat)

        # Singular value decomposition as per lemma 1 of https://arxiv.org/pdf/0905.2635.pdf.
        U, _, V = np.linalg.svd(self.A, full_matrices=True)
        C = np.ones((self.D, ))
        C[self.D-1] = np.linalg.det(np.dot(U, V))

        # Calculate the rotation matrix using Eq. 9 of https://arxiv.org/pdf/0905.2635.pdf.
        self.R = np.transpose(np.dot(np.dot(U, np.diag(C)), V))
        # Update scale and translation using Fig. 2 of https://arxiv.org/pdf/0905.2635.pdf.
        if self.scale:
            self.s = np.trace(np.dot(np.transpose(self.A),
                                     np.transpose(self.R))) / self.YPY
        else:
            pass
        if self.translate:
            self.t = np.transpose(muX) - self.s * \
                np.dot(np.transpose(self.R), np.transpose(muY))
        else:
            pass
            
from collections.abc import Iterable
def generate_pointcloud_and_indicator(meshes: Iterable[trimesh.Trimesh], strip:int=10) -> tuple:
    group_indicator = []
    point_cloud = []
    for mesh in meshes:
        if isinstance(mesh, list): # list of meshes -- combine them
            vertices = np.concatenate([m.vertices[::strip, :] for m in mesh], axis=0)
        elif isinstance(mesh, trimesh.Trimesh):
            vertices = mesh.vertices[::strip, :]
        group_indicator.append(len(vertices))
        point_cloud.append(vertices)
    point_cloud = np.concatenate(point_cloud, axis=0)
    return point_cloud, group_indicator



from tqdm import tqdm
class RealtimePlotCallback:
    def __init__(self, total_iterations):
        self.errors = [] 
        self.iterations = []
        self.progress_bar = tqdm(total=total_iterations, desc="Progress", unit="iter")

    def __call__(self, iteration, error, **kwargs):
        
        self.errors.append(error)
        self.iterations.append(iteration)

        self.progress_bar.update(1) 
        self.progress_bar.set_postfix({"error": f"{error:.4f}"})  
    
    def show_error_curve(self):
        plt.plot(self.iterations, self.errors)
        plt.xlabel("Iteration")
        plt.ylabel("Error")
        plt.show()

# target_group_indicator = [2, 3, 4]
# source_group_indicator = [1, 2, 3]
# group_mask = np.zeros((sum(target_group_indicator), sum(source_group_indicator)))
# arr1 = np.insert(np.cumsum(target_group_indicator), 0, 0)
# arr2 = np.insert(np.cumsum(source_group_indicator), 0, 0)
# for i, j, k, l in zip(arr1[0:-1], arr1[1:], arr2[0:-1], arr2[1:]):
#     group_mask[np.ix_(np.arange(i, j), np.arange(k, l))] = 1

# plt.imshow(group_mask)


class TreeGroupRigidRegistration(GroupRigidRegistration):
    def __init__(self, S1, S2, structure_lam=1, *args, **kwargs):
        '''
            S1: similarity matrix of the target point cloud
            S2: similarity matrix of the source point cloud
            structure_lam: the weight of the structure constraint
        '''
        super().__init__(*args, **kwargs)
        self.S1 = S1
        self.S2 = S2
        self.structure_lam = structure_lam
    
    def expectation(self):
        P = np.sum((self.X[None, :, :] - self.TY[:, None, :]) ** 2, axis=2)

        c = (2 * np.pi * self.sigma2) ** (self.D / 2)
        c = c * self.w / (1 - self.w)
        c = c * self.Ms / self.Ns # change this to group-wise
        gc = np.zeros((1, P.shape[1]), dtype=float)
        arr1 = np.insert(np.cumsum(self.target_group_indicator), 0, 0)
        for i, c_value in enumerate(c):
            start = arr1[i]
            end = arr1[i+1]
            gc[:, start:end] = c_value

        P = np.exp(-P / (2 * self.sigma2))
        P = P * self.group_mask # mask all irrelevant point-point connections
        den = np.sum(P, axis = 0, keepdims = True) # (1, N)
        den = np.clip(den, np.finfo(self.X.dtype).eps, None) + gc # add the group-wise constant

        self.P = np.divide(P, den)
        self.Pt1 = np.sum(self.P, axis=0)
        self.P1 = np.sum(self.P, axis=1)
        self.Np = np.sum(self.P1)
        self.PX = np.matmul(self.P, self.X)

        # structure constraint
        self.S = self.S2 @ self.P @ self.S1.T

    def update_transform(self):
        """
        Calculate a new estimate of the rigid transformation.

        """
        if self.translate:
            # target point cloud mean
            muX = np.divide(np.sum(self.PX, axis=0), self.Np) # updated 0117: use PX instead of computing it again
            # source point cloud mean
            muY = np.divide(
                np.sum(np.dot(np.transpose(self.P), self.Y), axis=0), self.Np)

            self.X_hat = self.X - np.tile(muX, (self.N, 1))
            # centered source point cloud
            Y_hat = self.Y - np.tile(muY, (self.M, 1))
        else: # no translation
            self.X_hat = self.X
            Y_hat = self.Y # the subtle part when there is no translation.

        self.YPY = np.dot(np.transpose(self.P1), np.sum(
            np.multiply(Y_hat, Y_hat), axis=1))

        if np.max(self.P) > 0.1: 
            self.A = self.X_hat.T @ (self.P.T + self.structure_lam*self.S.T) @ Y_hat
        else:
            self.A = self.X_hat.T @ self.P.T @ Y_hat

        # Singular value decomposition as per lemma 1 of https://arxiv.org/pdf/0905.2635.pdf.
        U, _, V = np.linalg.svd(self.A, full_matrices=True)
        C = np.ones((self.D, ))
        C[self.D-1] = np.linalg.det(np.dot(U, V))

        # Calculate the rotation matrix using Eq. 9 of https://arxiv.org/pdf/0905.2635.pdf.
        self.R = np.transpose(np.dot(np.dot(U, np.diag(C)), V))
        # Update scale and translation using Fig. 2 of https://arxiv.org/pdf/0905.2635.pdf.
        if self.scale:
            self.s = np.trace(np.dot(np.transpose(self.A),
                                     np.transpose(self.R))) / self.YPY
        else:
            pass
        if self.translate:
            self.t = np.transpose(muX) - self.s * \
                np.dot(np.transpose(self.R), np.transpose(muY))
        else:
            pass


class TreeGroupRigidRegistration2(GroupRigidRegistration):
    def __init__(self, S1, S2, structure_lam=1, *args, **kwargs):
        '''
            S1: similarity matrix of the target point cloud
            S2: similarity matrix of the source point cloud
            structure_lam: the weight of the structure constraint
        '''
        super().__init__(*args, **kwargs)
        self.S1 = S1
        self.S2 = S2
        self.structure_lam = structure_lam
    
    def expectation(self):
        P = np.sum((self.X[None, :, :] - self.TY[:, None, :]) ** 2, axis=2)

        c = (2 * np.pi * self.sigma2) ** (self.D / 2)
        c = c * self.w / (1 - self.w)
        c = c * self.Ms / self.Ns # group-wise handling
        gc = np.zeros((1, P.shape[1]), dtype=float)
        arr1 = np.insert(np.cumsum(self.target_group_indicator), 0, 0)
        for i, c_value in enumerate(c):
            start = arr1[i]
            end = arr1[i+1]
            gc[:, start:end] = c_value

        P = np.exp(-P / (2 * self.sigma2))
        P = P * self.group_mask # mask all irrelevant point-point connections
        den = np.sum(P, axis = 0, keepdims = True) # (1, N)
        den = np.clip(den, np.finfo(self.X.dtype).eps, None) + gc 

        self.P = np.divide(P, den)
        self.Pt1 = np.sum(self.P, axis=0) # Column sum
        self.P1 = np.sum(self.P, axis=1)  # Row sum
        self.Np = np.sum(self.P1)
        self.PX = np.matmul(self.P, self.X) 

        # Structure constraint
        self.S = self.S2 @ self.P @ self.S1.T

    def update_transform(self):
        """
        Calculate a new estimate of the rigid transformation using combined weights.
        """
        use_structure = np.max(self.P) > 0.1 # 0.01 # used to be 0.1
        # print(use_structure, np.max(self.P))
        if use_structure:
            self.W_matrix = self.P + self.structure_lam * self.S
            self.W_row_sum = self.P1 + self.structure_lam * np.sum(self.S, axis=1)
            self.W_col_sum = self.Pt1 + self.structure_lam * np.sum(self.S, axis=0)
            self.N_eff = np.sum(self.W_row_sum) # N_W

            # W @ X = P @ X + lam * (S @ X)
            PX_eff = self.PX + self.structure_lam * np.dot(self.S, self.X)
            # W^T @ Y = P^T @ Y + lam * (S^T @ Y)
            PTY_eff = np.dot(self.P.T, self.Y) + self.structure_lam * np.dot(self.S.T, self.Y)

        else:
            self.W_matrix = self.P
            self.W_row_sum = self.P1
            self.W_col_sum = self.Pt1
            self.N_eff = self.Np
            
            PX_eff = self.PX
            PTY_eff = np.dot(self.P.T, self.Y)

        if self.translate:
            muX = np.divide(np.sum(PX_eff, axis=0), self.N_eff) 
            muY = np.divide(np.sum(PTY_eff, axis=0), self.N_eff)

            self.X_hat = self.X - np.tile(muX, (self.N, 1))
            Y_hat = self.Y - np.tile(muY, (self.M, 1))
        else:
            self.X_hat = self.X
            Y_hat = self.Y 
        self.YPY = np.dot(np.transpose(self.W_row_sum), np.sum(np.multiply(Y_hat, Y_hat), axis=1))

        self.A = self.X_hat.T @ self.W_matrix.T @ Y_hat

        U, _, V = np.linalg.svd(self.A, full_matrices=True)
        C = np.ones((self.D, ))
        C[self.D-1] = np.linalg.det(np.dot(U, V))
        self.R = np.transpose(np.dot(np.dot(U, np.diag(C)), V))

        if self.scale:
            self.s = np.trace(np.dot(np.transpose(self.A), np.transpose(self.R))) / self.YPY
        
        if self.translate:
            self.t = np.transpose(muX) - self.s * np.dot(np.transpose(self.R), np.transpose(muY))

    def update_variance(self):
            """
            Update the variance of the mixture model using the new estimate of the rigid transformation.
            Incorporates structural constraints (W = P + \lambda S) by using the combined weights.
            """
            qprev = self.q

            trAR = np.trace(np.dot(np.transpose(self.A), self.R))

            xPx = np.dot(np.transpose(self.W_col_sum), np.sum(np.multiply(self.X_hat, self.X_hat), axis=1))

            self.q = (xPx - 2 * self.s * trAR + self.s * self.s * self.YPY) / \
                (2 * self.sigma2) + self.D * self.N_eff / 2 * np.log(self.sigma2)
            
            self.diff = np.abs(self.q - qprev)

            self.sigma2 = (xPx - self.s * trAR) / (self.N_eff * self.D)

            if self.sigma2 <= 0:
                self.sigma2 = self.tolerance / 10