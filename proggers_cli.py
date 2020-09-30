#!/usr/bin/python3
# WIP commandline client

from cmd import Cmd

import proggers as pv

src = pv.SomeSource('...')
prog = pv.Progger(source=src, result_cb=print)
q = pv.Query()  # or where='...', group_py='...'
q.where('...')
q.group_by('...')
prog.execute(q, result_cb=print)


class Client(Cmd):
    def do_group_by(self, line):
        pass

    def do_where(self, line):
        pass

    def do_stop(self, line):
        pass

    def do_start(self, line):
        pass


if __name__ == '__main__':
    Client().cmdloop()
