from src.dataloader import DataLoader
from src.r0_generator import R0Generator


def main():
    data = DataLoader()
    r0 = R0Generator(param=data.model_parameters_data, n_age=16)


if __name__ == '__main__':
    main()
