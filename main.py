from src.runner import Runner
from src.static.dataloader import DataLoader


def main():
    model = "rost_agg"
    target = "r0"  # "r0", "hospitalized", "icu", "death"
    # use elasticity i.e, ngm or cm or none
    use_ngm_elasticity = False
    use_cm_elasticity = True

    data = DataLoader(model=model)
    runner = Runner(data=data, model=model, target=target,
                    use_ngm_elasticity=use_ngm_elasticity,
                    use_cm_elasticity=use_cm_elasticity)

    runner.run()


if __name__ == '__main__':
    main()
