def generate(args, device):

    ## generate env
    if args.env == 'Dobot-v1':
        from myrllib.envs.dobot_env import DobotEnv_V1
        env = DobotEnv_V1()
    elif args.env == 'Dobot-v2':
        from myrllib.envs.dobot_env import DobotEnv_V2
        env = DobotEnv_V2()
    elif args.env == 'Dobot-v3':
        from myrllib.envs.dobot_env import DobotEnv_V3
        env = DobotEnv_V3()
    elif args.env == 'Dobot-v4':
        from myrllib.envs.dobot_env import DobotEnv_V4
        env = DobotEnv_V4()
    else:
        raise Exception("No exist env!")

    ## generate learner
    from myrllib.algorithms.ddpg import DDPG
    learner = DDPG(args=args,
                   state_dim=env.state_dim,
                   action_dim=env.action_dim,
                   max_action=float(env.action_bound[1]),
                   goal_dim=env.goal_dim,
                   device=device)

    ## generate shaper
    if args.shaper == 'pbrs':
        from myrllib.shapers.pbrs import PBRS
        shaper = PBRS(args, env)
    elif args.shaper == 'dpba':
        from myrllib.shapers.dpba import DPBA
        shaper = DPBA(args, env, device)
    elif args.shaper == 'mfrs':
        from myrllib.shapers.mfrs import MFRS
        shaper = MFRS(args, env, device)
    elif args.shaper == 'aim':
        from myrllib.shapers.aim import AIM
        shaper = AIM(args, device)
    else:
        shaper = None

    return env, learner, shaper