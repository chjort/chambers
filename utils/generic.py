import inspect


def deserialize_object(identifier, module_objects, module_name, **kwargs):
    if type(identifier) is str:
        obj = module_objects.get(identifier)
        if obj is None:
            raise ValueError('Unknown ' + module_name + ':' + identifier)
        if inspect.isclass(obj):
            obj = obj(**kwargs)
        elif callable(obj):
            obj = obj(**kwargs)
        return obj

    else:
        raise ValueError('Could not interpret serialized ' + module_name +
                         ': ' + identifier)