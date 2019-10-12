import collections
import logging
import types

logger = logging.getLogger(__name__)

class ConnectionInfo(types.SimpleNamespace):
    """
        Stores info about a *connection*, or the joining of a Projection to a Port

        **compositions** : the `Composition`\\ s which the connection is associated with
        **active_context** : the `ContextFlags` under which the connection is active
    """

    ALL = True

    def __init__(self, compositions=None, active_context=None):
        if compositions is not None and compositions is not self.ALL:
            if isinstance(compositions, collections.abc.Iterable):
                compositions = set(compositions)
            else:
                compositions = {compositions}

        super().__init__(compositions=compositions, active_context=active_context)

    def add_composition(self, composition):
        if self.compositions is self.ALL:
            logger.info('Attempted to add composition to {} but is set to ConnectionInfo.ALL'.format(self))
        elif self.compositions is None:
            if composition is self.ALL:
                self.compositions = composition
            else:
                self.compositions = {composition}
        else:
            self.compositions.add(composition)

    def is_active_in_composition(self, composition):
        if self.compositions is None:
            return False

        if self.compositions is self.ALL:
            return True

        return composition in self.compositions
