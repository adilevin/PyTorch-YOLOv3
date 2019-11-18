from __future__ import division

original_attribs = [
    'size', 'stride', 'pad', 'activation', 'batch_normalize'
]


added_attribs = [
    'index', 'type', 'cum_stride', 'filters', 'from'
]

mapped_names = {
    'convolutional': 'conv',
    'shortcut': 'add',
    'route': 'route',
    'yolo': 'YOLO',
    'upsample': 'up'
}


def get_or_default(obj, attr, default):
    if attr in obj.keys():
        return obj[attr]
    else:
        return default


def csv_header():
    return ','.join(added_attribs + original_attribs)


def csv_line(module_def, **additional_def):
    attr = [get_or_default(additional_def, a, '') for a in added_attribs] + [get_or_default(module_def, a, '') for a in original_attribs]
    return ','.join(map(str, attr))


def generate_csv(module_defs):
    hyperparams = module_defs.pop(0)
    output_filters = [int(hyperparams["channels"])]
    cumulative_stride = [1]
    yield csv_header()
    for module_i, module_def in enumerate(module_defs):
        added_attribs = {
            'index': module_i,
            'type': mapped_names[module_def['type']]
        }
        if module_def["type"] == "convolutional":
            filters = int(module_def["filters"])
            cum_stride = cumulative_stride[-1] * int(module_def["stride"])
        elif module_def["type"] == "maxpool":
            filters = output_filters[-1]
            cum_stride = cumulative_stride[-1] * int(module_def["stride"])
        elif module_def["type"] == "upsample":
            filters = output_filters[-1]
            cum_stride = cumulative_stride[-1] / int(module_def["stride"])
        elif module_def["type"] == "route":
            layers = [int(x) for x in module_def["layers"].split(",")]
            added_attribs['from'] = f'[ {" ".join(map(str, layers))}]'
            filters = sum([output_filters[1:][i] for i in layers])
            cum_stride = cumulative_stride[layers[0]]
        elif module_def["type"] == "shortcut":
            filters = output_filters[1:][int(module_def["from"])]
            added_attribs['from'] = f'[ -1 {module_def["from"]}]'
            cum_stride = cumulative_stride[-1]
        elif module_def["type"] == "yolo":
            filters = int(module_def["classes"]) + 5
            cum_stride = cumulative_stride[-1]
        output_filters.append(filters)
        cumulative_stride.append(cum_stride)
        added_attribs['filters'] = filters
        added_attribs['cum_stride'] = cum_stride
        yield csv_line(module_def=module_def, **added_attribs)

if __name__ == "__main__":
    from models import parse_model_config
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_def", type=str, default="config/yolov3.cfg", help="path to model definition file")
    parser.add_argument("--batch_size", type=int, default=1, help="size of the batches")
    parser.add_argument("--img_size", type=int, default=416, help="size of each image dimension")
    opt = parser.parse_args()
    module_defs_ = parse_model_config(opt.model_def)
    csv = [line for line in generate_csv(module_defs_)]
    print('\n'.join(csv))
