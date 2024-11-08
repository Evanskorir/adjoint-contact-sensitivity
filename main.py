from src.runner import Runner
from src.static.dataloader import DataLoader


def main():
    data = DataLoader(model="kenya")
    runner = Runner(data=data, model="kenya")
    runner.run()


if __name__ == '__main__':
    main()
