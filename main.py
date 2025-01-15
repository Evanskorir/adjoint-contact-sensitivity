from src.runner import Runner
from src.static.dataloader import DataLoader


def main():
    model = "rost"
    data = DataLoader(model=model)
    runner = Runner(data=data, model=model)
    runner.run()


if __name__ == '__main__':
    main()
