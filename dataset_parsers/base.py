
import abc


class Dataset(metaclass=abc.ABCMeta):
    @classmethod
    def __subclasshook__(cls, subclass):
        return (hasattr(subclass, 'get_file') and 
                callable(subclass.get_file) and 
                hasattr(subclass, 'get_label') and 
                callable(subclass.get_label) and
                hasattr(subclass, 'to_self_supervised') and 
                callable(subclass.to_self_supervised))
