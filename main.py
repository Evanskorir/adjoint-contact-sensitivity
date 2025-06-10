from src.runner import Runner
from src.static.dataloader import DataLoader


def main():
    model = "rost"
    target = "death"  # "r0", "hospitalized", "icu", "death"
    use_cm_elasticity = False

    data = DataLoader(model=model)
    runner = Runner(data=data, model=model, target=target,
                    use_cm_elasticity=use_cm_elasticity)

    runner.run()


if __name__ == '__main__':
    main()
