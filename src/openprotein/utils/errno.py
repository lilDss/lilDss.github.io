from string import Template

class Error(Exception):
    def __init__(self, args):
        # self.what = what
        # self.code = code
        # msg = what
        # if code:
        #     msg = '%s: %s' % (what, self.reason)
        #     hint = getattr(self, 'MDB_HINT', None)
        #     if hint:
        #         msg += ' (%s)' % (hint,)
        TEM = getattr(self, "TEM", None)
        msg = TEM.substitute(msg=args)
        Exception.__init__(self, msg)

class FileNotFoundError(Error):
    TEM = Template('No such file or directory: $msg')
