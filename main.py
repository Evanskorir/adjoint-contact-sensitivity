from src.runner import Runner
from src.static.dataloader import DataLoader


def main():
    model = "british_columbia"
    method = "svd"
    data = DataLoader(model=model)
    runner = Runner(data=data, model=model, method=method)
    runner.run()


if __name__ == '__main__':
    main()
