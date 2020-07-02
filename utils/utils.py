import os
from glob import glob


def find_model(path, epoch='latest'):
    if epoch == 'latest':
        files = glob(os.path.join(path, '*.pth'))
        file = sorted(files, key=lambda x: int(x.rsplit('.', 2)[1]))[-1]
    else:
        file = os.path.join(path, 'weights.{:d}.pth'.format(int(epoch)))
    assert os.path.exists(file), 'File not found: ' + file
    print('Find model of {} epoch: {}'.format(epoch, file))
    return file


def check_attribute_conflict(att_batch, att_name, att_names):
    def _get(att, att_name):
        if att_name in att_names:
            return att[att_names.index(att_name)]
        else:
            return None

    def _set(att, value, att_name):
        if att_name in att_names:
            att[att_names.index(att_name)] = value

    att_id = att_names.index(att_name)

    for att in att_batch:
        if att_name in ['Bald', 'Receding_Hairline'] and att[att_id] != 0:
            if _get(att, 'Bangs') != 0:
                _set(att, 1-att[att_id], 'Bangs')
        elif att_name == 'Bangs' and att[att_id] != 0:
            for n in ['Bald', 'Receding_Hairline']:
                if _get(att, n) != 0:
                    _set(att, 1-att[att_id], n)
                    _set(att, 1-att[att_id], n)
        elif att_name in ['Black_Hair', 'Blond_Hair',
                          'Brown_Hair', 'Gray_Hair'] and att[att_id] != 0:
            for n in ['Black_Hair', 'Blond_Hair', 'Brown_Hair', 'Gray_Hair']:
                if n != att_name and _get(att, n) != 0:
                    _set(att, 1-att[att_id], n)
        elif att_name in ['Straight_Hair', 'Wavy_Hair'] and att[att_id] != 0:
            for n in ['Straight_Hair', 'Wavy_Hair']:
                if n != att_name and _get(att, n) != 0:
                    _set(att, 1-att[att_id], n)
        elif att_name in ['Mustache', 'No_Beard'] and att[att_id] != 0:
            for n in ['Mustache', 'No_Beard']:
                if n != att_name and _get(att, n) != 0:
                    _set(att, 1-att[att_id], n)

    return att_batch


def name_experiment(prefix='', suffix=''):
    import datetime
    import platform
    plt = platform.node()
    experiment_name = datetime.datetime.now().strftime('%b%d_%H-%M-%S_') + plt

    if prefix is not None and prefix != '':
        experiment_name = prefix + '_' + experiment_name
    if suffix is not None and suffix != '':
        experiment_name = experiment_name + '_' + suffix
    return experiment_name


class Progressbar():
    def __init__(self):
        self.p = None

    def __call__(self, iterable):
        from tqdm import tqdm
        self.p = tqdm(iterable)

        return self.p

    def say(self, **kwargs):
        if self.p is not None:
            self.p.set_postfix(**kwargs)


def add_scalar_dict(writer, scalar_dict, iteration, directory=None):
    for key in scalar_dict:
        key_ = directory + '/' + key if directory is not None else key
        writer.add_scalar(key_, scalar_dict[key], iteration)
