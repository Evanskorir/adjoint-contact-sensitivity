from src.runner import Runner
from src.static.dataloader import DataLoader


def main():
    data = DataLoader(model="moghadas")
    runner = Runner(data=data, model="moghadas")
    runner.run()


if __name__ == '__main__':
    main()
