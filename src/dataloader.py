import json
import os
import torch
import xlrd

PROJECT_PATH = os.path.dirname(os.path.realpath(__file__))


class DataLoader:
    def __init__(self):
        self._model_parameters_data_file = os.path.join(PROJECT_PATH, "../data",
                                                        "model_parameters.json")
        self._contact_data_file = os.path.join(PROJECT_PATH, "../data",
                                               "contact_matrices.xls")
        self._age_data_file = os.path.join(PROJECT_PATH, "../data",
                                           "age_distribution.xls")

        self._get_age_data()
        self._get_model_parameters_data()
        self._get_contact_mtx()

    def _get_age_data(self):
        wb = xlrd.open_workbook(self._age_data_file)
        sheet = wb.sheet_by_index(0)
        datalist = torch.tensor([sheet.row_values(i) for i in range(0, sheet.nrows)],
                                dtype=torch.float32)
        wb.unload_sheet(0)
        self.age_data = datalist

    def _get_model_parameters_data(self):
        # Load model parameters
        with open(self._model_parameters_data_file) as f:
            parameters = json.load(f)
        self.model_parameters_data = dict()
        for param in parameters.keys():
            param_value = parameters[param]["value"]
            if isinstance(param_value, list):
                self.model_parameters_data[param] = torch.tensor(param_value,
                                                                 dtype=torch.float32)
            else:
                self.model_parameters_data[param] = param_value

    def _get_contact_mtx(self):
        wb = xlrd.open_workbook(self._contact_data_file)
        contact_matrices = dict()
        flattened_vectors = dict()
        for idx in range(4):
            sheet = wb.sheet_by_index(idx)
            datalist = torch.tensor([sheet.row_values(i) for i in range(0, sheet.nrows)],
                                    dtype=torch.float32)
            cm_type = wb.sheet_names()[idx]
            wb.unload_sheet(0)
            datalist = self.transform_matrix(datalist)
            contact_matrices[cm_type] = datalist

            # Get the upper triangle elements, including the diagonal
            upper_tri_idx = torch.triu_indices(datalist.size(0), datalist.size(1),
                                               offset=0)
            upper_tri_elem = datalist[upper_tri_idx[0], upper_tri_idx[1]]
            # Set requires_grad to True
            upper_tri_elem.requires_grad = True
            # Save upper triangle elements in the dictionary
            flattened_vectors[cm_type] = upper_tri_elem

            # Save the transformed flattened vectors
        self.flattened_vector_dict = flattened_vectors

        # Save the transformed matrices
        self.contact_data = contact_matrices

    def transform_matrix(self, matrix: torch.Tensor):
        # Get age vector as a column vector
        age_distribution = self.age_data.reshape((-1, 1))   # (16, 1)
        # Get matrix of total number of contacts
        matrix_1 = matrix * age_distribution        # (16, 16)
        # Get symmetrized matrix
        output = (matrix_1 + matrix_1.t()) / 2
        # Get contact matrix
        output /= age_distribution   # divides and assign the result to output    (16, 16)
        return output
