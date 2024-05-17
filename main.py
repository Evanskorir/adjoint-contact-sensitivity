import torch
from src.dataloader import DataLoader

from src.r0_generator import R0Generator
from src.upper_tri_elements import MatrixOperations


def main():
    data = DataLoader()
    sim = R0Generator(param=data.model_parameters_data, n_age=16)
    # Create the contact matrix
    C_mtx = (
            data.contact_data["Home"]
            + data.contact_data["School"]
            + data.contact_data["Work"]
            + data.contact_data["Other"]
    )

    # Create a tensor from C_mtx and set it to require gradients
    contact_mtx = torch.tensor(C_mtx, dtype=torch.float32, requires_grad=True)

    # Get upper triangle elements
    Matr = MatrixOperations(matrix=contact_mtx, n_age=16)
    # get matrix from upper tri elements
    upper_tri_elem = Matr.get_upper_triangle_elements()

    # Construct matrix from upper triangle elements
    reconstructed_matrix = Matr.upper_triangle_to_matrix(upper_tri_elem=upper_tri_elem)

    print("Original Contact Matrix:", C_mtx)
    print("Upper Triangle Elements:", upper_tri_elem)
    print("Reconstructed Matrix:", reconstructed_matrix)

    susceptibility = torch.ones(16)
    for susc in torch.tensor([0.5]):
        susceptibility[:4] = susc
        data.model_parameters_data.update({"susc": susceptibility})
        dominant_eig_val, ngm_small_tensor, \
        ngm_small_grads = sim.get_eig_val(contact_mtx=contact_mtx)

        print("Dominant eigenvalue:", dominant_eig_val)
        print("NGM small tensor:", ngm_small_tensor)
        print("Gradients:", ngm_small_grads)


if __name__ == '__main__':
    main()
