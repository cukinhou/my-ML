from abc import ABCMeta, abstractmethod


class Classifier(object):
    """
    Abstract base class for implementing a classifier.
    """
    __metaclass__ = ABCMeta

    @classmethod
    def load(cls, infile):
        '''
        This method un-pickles a saved Classifier object.
        Parameters
        ----------
        :param infile: name of Pickled classifier
        :type infile: str
        :return: classifier object
        :rtype: Classifier
        '''
        import pickle
        import os

        assert os.path.exists(infile), 'Cannot find path: {0}'.format(infile)

        # close the open file if needed and use its name
        try:
            infile.close()
            infile = infile.name
        except AttributeError:
            pass
        # instantiate a new Processor and return it
        with open(infile, 'rb') as f:
            # Python 2 and 3 behave differently
            try:
                # Python 3
                obj = pickle.load(f, encoding='latin1')
            except TypeError:
                # Python 2 doesn't have/need the encoding
                obj = pickle.load(f)
        # warn if the unpickled Processor is of other type
        if obj.__class__ is not cls:
            import warnings
            warnings.warn("Expected Processor of class '%s' but loaded "
                          "Processor is of class '%s', processing anyways." %
                          (cls.__name__, obj.__class__.__name__))
        return obj

    def dump(self, name='default.pkl', path='models/'):
        """
        This method pickles a Classifier object and saves it.
        :param name: name of the output pickle file
        :type name: str
        :param path: folder where to save the file
        :type path: str
        """
        import os
        import pickle
        assert os.path.exists(path), 'Cannot find path: {0}'.format(path)
        outfile = path + name
        # close the open file if needed and use its name
        try:
            outfile.close()
            outfile = outfile.name
        except AttributeError:
            pass
        # dump the Processor to the given file
        # Note: for Python 2 / 3 compatibility reason use protocol 2
        pickle.dump(self, open(outfile, 'wb'), protocol=2)

    @abstractmethod
    def predict(self, data, **kwargs):
        """
        This method must be implemented by the derived class and should
        predict an output for the given data.
        Parameters
        ----------
        data : depends on the implementation of subclass
            Data to be processed.
        kwargs : dict, optional
            Keyword arguments for predicting.
        Returns
        -------
        depends on the implementation of subclass
            Processed data.
        """
        return

    @abstractmethod
    def fit(self, data, **kwargs):
        """
        This method must be implemented by the derived class and should
        fit the classifier model to the input data.
        Parameters
        ----------
        data : depends on the implementation of subclass
            Data to be processed.
        kwargs : dict, optional
            Keyword arguments for processing.
        Returns
        -------
        depends on the implementation of subclass.
        """
        return

    def __call__(self, *args, **kwargs):
        # makes Classifier callable
        return self.predict(*args, **kwargs)
