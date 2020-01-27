def create_model(opt):
    model = opt['model']

    if model == 'srragan':
        from .SRRaGAN_model import SRRaGANModel as M
    elif model == 'srim':
        from .SRIM_model import SRIMModel as M
    elif model == 'srim_mc':
        from .SRIM_mult_code_model import SRIMMultCodeModel as M
    else:
        raise NotImplementedError('Model [{:s}] not recognized.'.format(model))
    m = M(opt)
    print('Model [{:s}] is created.'.format(m.__class__.__name__))
    return m
