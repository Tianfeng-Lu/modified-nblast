import navis
from GroupRigidRegistration import GroupRigidRegistration, TreeGroupRigidRegistration, TreeGroupRigidRegistration2
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import trimesh
import ast

## utils ##
def _set_coords(n, new_co):
    """Set new xyz coordinates for a tree neuron (from navis.transforms.align)."""
    if new_co.ndim == 2 and new_co.shape[1] == 1:
        new_co = new_co.flatten()
    if not isinstance(n, navis.TreeNeuron):
        raise TypeError(f'Unable to extract coordinates from {type(n)}')
    if new_co.ndim == 2:
            n.nodes[['x', 'y', 'z']] = new_co
    # If this is a single vector
    else:
        for i in 'xyz':
            n.nodes[i] = new_co


def compute_node_similarity(n, sigma: float | str = "auto", normalization=True):
    """
        Given a tree neuron, the function computes the node adjacency as a structure constraint.
        parameters
            sigma: the variance of the Gaussian kernel, controls the range of the structure constraint. 
                A smaller value means that only nearby nodes will have a strong similarity, while a larger sigma allows for more distant nodes to be considered similar. 
                default is "auto", which sets sigma to the 90th percentile of the minimum non-zero distances between nodes, ensuring that most nodes have at least one neighbor with a significant similarity.
            normalization: whether to row-normalize the similarity matrix, default is True.
        returns
            S: the node similarity matrix, shape (#nodes, #nodes)
    """
    distance_matrix = navis.geodesic_matrix(n, directed=False).to_numpy()
    if sigma == "auto":
        tmp = distance_matrix.copy()
        np.fill_diagonal(tmp, np.inf)
        sigma = np.quantile(tmp.min(axis=1), 0.9)

    S = np.exp(-(distance_matrix**2) / (2 * sigma**2))
    if normalization:
        row_sums = S.sum(axis=1, keepdims=True)
        S = S / (row_sums + 1e-9)
    return S


def generate_group_indicator(label, label_order=["soma", "cellbodyfiber", "dendrite", "linker", "axon"]):
    """
        Given labels corresponding to a point cloud, generate a group indicator, index mapping for the TreeGroupRigidRegistration objects.
        That's due to the current group indicators are list of integers indicating the length of each group to be paired, so we need to sort the point cloud according to the label order, and then generate the group indicator and index mapping accordingly.
        parameters
            label: the corresponding label for each point, shape (N,)
            label_order: the order of the labels to be sorted, default is ["soma", "cellbodyfiber", "dendrite", "linker", "axon"]
        returns
            group_indicator: list of integers indicating the length of each group to be paired.
            index_mapping: a dictionary mapping to map back the sorted indices to the original indices of the point cloud.
    """
    label = np.array(label)
        
    sorted_indices = []
    group_indicator = []
    
    for l in label_order:
        indices = np.where(label == l)[0]
        group_indicator.append(len(indices))
        sorted_indices.append(indices)
        
    index_mapping = np.concatenate(sorted_indices)
    
    return group_indicator, index_mapping
    
def centering_neuron(n, inplace=False):
    if not inplace:
        n = n.copy()
    if getattr(n, 'soma_pos', None) is None:
        n.soma_pos = n.nodes[n.nodes["label"]=="soma"].iloc[0][["x","y","z"]].to_numpy().astype(float)
    n.nodes[["x","y","z"]] = (n.nodes[["x","y","z"]] - n.soma_pos) # place the soma to the origin
    return n

## alignment ##
def align_neuron_to_template(n, target_point_cloud, s1, *, structure_lam=1, w=0, max_iterations=50, tolerance=1e-2):
    n = n.copy()
    n.nodes[["x","y","z"]] = (n.nodes[["x","y","z"]] - n.soma_pos) # place the soma to the origin
    source_point_cloud = n.nodes[["x","y","z"]].to_numpy()
    if hasattr(n, 'S'):
        s2 = n.S
    else:
        s2 = compute_node_similarity(n, sigma="auto")
    regis = TreeGroupRigidRegistration2(X=target_point_cloud, Y=source_point_cloud, S1=s1, S2=s2, structure_lam=structure_lam, w=w, scale=False, translate=False, max_iterations=max_iterations, tolerance=tolerance)
    _ = regis.register()
    regis.transform_point_cloud()
    _set_coords(n, regis.TY)
    return n


def align_neuron_to_template_with_compartments(n, target_point_cloud_sorted, s1_sorted, target_group_indicator, *, structure_lam=1, w=0, max_iterations=50, tolerance=1e-2):
    n = n.copy()
    source_point_cloud = n.nodes[["x","y","z"]].to_numpy()
    s2 = compute_node_similarity(n, sigma="auto")
    labels = n.nodes["label"].to_numpy()

    source_group_indicator, index_mapping = generate_group_indicator(labels)
    source_point_cloud_sorted = source_point_cloud[index_mapping]
    s2_sorted = s2[index_mapping][:, index_mapping]

    regis = TreeGroupRigidRegistration2(X=target_point_cloud_sorted, Y=source_point_cloud_sorted, source_group_indicator=source_group_indicator, target_group_indicator=target_group_indicator, S1=s1_sorted, S2=s2_sorted, structure_lam=structure_lam, w=w, scale=False, translate=False, max_iterations=max_iterations, tolerance=tolerance)
    _ = regis.register()
    regis.transform_point_cloud()
    # keep the spatial coordinates of the axon unchanged, and only update the coordinates of the soma, cellbodyfiber, dendrite and linker according to the registration result.
    _set_coords(n, regis.TY[np.argsort(index_mapping)])
    return n

## NBLAST ##
from typing import Literal
def rigid_invariant_nblast(query: navis.TreeNeuron|navis.NeuronList, target: navis.TreeNeuron|navis.NeuronList=None, *, template: Literal["auto", "pairwise"] | int | navis.TreeNeuron = "auto", **kwargs):
    """
        Rigid-invariant NBLAST score computation.
        parameters
            template: the template for alignment, can be "auto", "pairwise", an integer indicating the index of the neuron in the target set, or a TreeNeuron object. If it's set to "auto", the function will randomly select a few neurons from the target set as templates. If it's set to "pairwise", the function will compute the NBLAST score score after aligning each pair.
            kwargs: additional keyword arguments for the NBLAST score computation, such as "smat", the score matrix to be used for NBLAST, default is "auto" 
    """
    if isinstance(query, navis.TreeNeuron):
        query = navis.NeuronList([query])
    if target is not None and isinstance(target, navis.TreeNeuron):
        target = navis.NeuronList([target])
    query.apply(centering_neuron, inplace=True)
    query.apply(lambda n: setattr(n, 'S', compute_node_similarity(n, sigma="auto")))
    if target is None:
        target = query
        all_by_all = True
    else:
        target.apply(centering_neuron, inplace=True)
        target.apply(lambda n: setattr(n, 'S', compute_node_similarity(n, sigma="auto")))
        all_by_all = False

    if template == "auto":
        step = max(1, len(target) // 10)
        template_neurons = target[::step][:10]
    elif template == "pairwise":
        template_neurons = target
    elif isinstance(template, int):
        template_neurons = [target[template]]
    elif isinstance(template, navis.TreeNeuron):
        template_neurons = [template]
    template_neurons = navis.NeuronList(template_neurons)
    template_neurons.apply(centering_neuron, inplace=True)
    template_neurons.apply(lambda n: setattr(n, 'S', compute_node_similarity(n, sigma="auto")))

    res_df = None
    for template_neuron in template_neurons:
        if not np.allclose(template_neuron.soma_pos, np.zeros(3)):
            raise ValueError("Please use the centered neuron as the template.")
        s1 = template_neuron.S
        target_point_cloud = template_neuron.nodes[["x","y","z"]].to_numpy()
        # alignment
        aligned_neurons = []
        for n in query:
            n = align_neuron_to_template(n, target_point_cloud, s1, structure_lam=2)
            aligned_neurons.append(n)
        aligned_neurons = navis.NeuronList(aligned_neurons)

        # NBLAST
        if all_by_all:
            tmp_df = navis.nbl.nblast_allbyall(navis.make_dotprops(aligned_neurons), **kwargs)
        else:
            aligned_targets = []
            for target_neuron in target:
                target_neuron = align_neuron_to_template(target_neuron, target_point_cloud, s1, structure_lam=2)
                aligned_targets.append(target_neuron)
            aligned_targets = navis.NeuronList(aligned_targets)
            tmp_df= navis.nbl.nblast(navis.make_dotprops(aligned_neurons), navis.make_dotprops(aligned_targets), scores="max", **kwargs)
        if res_df is None:
            res_df = tmp_df
        else:
            res_df = np.maximum(res_df, tmp_df)
    
    return res_df

def rigid_invariant_nblast_compartment(query: navis.TreeNeuron|navis.NeuronList, target: navis.TreeNeuron|navis.NeuronList=None, *, template: Literal["auto", "pairwise"] | int | navis.TreeNeuron = "auto", **kwargs):
    """
        Rigid-invariant NBLAST score computation with compartment-aware alignment.
        requires the "label" column in the neuron nodes to indicate the compartment information. Especially, the axon will be held out from the alignment.
        parameters
            template: the template for alignment, can be "auto", "pairwise", an integer indicating the index of the neuron in the target set, or a TreeNeuron object. If it's set to "auto", the function will randomly select a few neurons from the target set as templates. If it's set to "pairwise", the function will compute the NBLAST score after aligning each pair.
            kwargs: additional keyword arguments for the NBLAST score computation, such as 
                "smat", the score matrix to be used for NBLAST, default is "auto";
        returns
            neurite_nblast_score: the NBLAST score for the neurite part (in neuropil)
            axon_nblast_score: the NBLAST score for the axon part
    """
    if isinstance(query, navis.TreeNeuron):
        query = navis.NeuronList([query])
    query_neurite = navis.NeuronList([navis.subset_neuron(n, n.nodes.query("label != 'axon'")["node_id"], inplace=False) for n in query])
    query_axon = navis.NeuronList([navis.subset_neuron(n, n.nodes.query("label == 'axon'")["node_id"], inplace=False) for n in query])
    if target is not None and isinstance(target, navis.TreeNeuron):
        target = navis.NeuronList([target])
    if target is not None:
        target_neurite = navis.NeuronList([navis.subset_neuron(n, n.nodes.query("label != 'axon'")["node_id"], inplace=False) for n in target]) # subset the neurite part
        target_axon = navis.NeuronList([navis.subset_neuron(n, n.nodes.query("label == 'axon'")["node_id"], inplace=False) for n in target]) # subset the axon part
    else:
        target_neurite = None
        target_axon = None
    
    neurite_nblast_score = rigid_invariant_nblast(query=query_neurite, target=target_neurite, template=template, **kwargs)
    if target is None:
        axon_nblast_score = navis.nbl.nblast_allbyall(navis.make_dotprops(query_axon), **kwargs)
    else:
        axon_nblast_score = navis.nbl.nblast(navis.make_dotprops(query_axon), target=navis.make_dotprops(target_axon), scores="max", **kwargs)

    return neurite_nblast_score, axon_nblast_score


## exporting ##
def plotlymesh_to_ppt(fig, filename):
    scene = trimesh.Scene()
    scene.bg_color = [30, 30, 30, 255]
    for i, trace in enumerate(fig.data):
        if trace.type == 'mesh3d':
            vertices = np.column_stack((trace.x, trace.y, trace.z))
            
            if trace.i is not None and trace.j is not None and trace.k is not None:
                faces = np.column_stack((trace.i, trace.j, trace.k))
                
                mesh = trimesh.Trimesh(vertices=vertices, faces=faces)
                
                if hasattr(trace, 'color'):
                    if isinstance(trace.color, str) and trace.color.startswith("rgb"):
                        rgb_tuple = ast.literal_eval(trace.color.strip("rgb"))
                    elif isinstance(trace.color, str) and trace.color.startswith("#"):
                        rgb_tuple = tuple(int(trace.color[i:i+2], 16) for i in (1, 3, 5))
                    else:
                        rgb_tuple = (255, 255, 255)  # default to white if color format is unrecognized
                    mesh.visual.face_colors = [int(max(0, min(255, c * 0.5))) for c in rgb_tuple]
                scene.add_geometry(mesh, node_name=f"mesh_{i}")

    scene.export(filename)
    print(f"file saved as: {filename}")