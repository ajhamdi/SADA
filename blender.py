from numbers import Number
from subprocess import Popen, PIPE
import numpy as np


class Blender(object):
	def __init__(self, py_file, blend_file=None, *args):
		self._args = list(args)
		self._blend_file = blend_file
		self._commands = []

		# read code and register all top-level non-hidden functions
		with open(py_file, 'r') as file:
			self._code = file.read() + '\n' * 2
			for line in self._code.splitlines():
				if line.startswith('def ') and line[4] != '_':
					self._register(line[4:line.find('(')])

	def __call__(self, blender_func_name, *args, **kwargs):
		args = str(args)[1:-1] + ',' if len(args) > 0 else ''
		kwargs = ''.join([k + '=' + repr(v) + ',' for k, v in kwargs.items()])
		cmd = blender_func_name + '(' + (args + kwargs)[:-1] + ')'
		self._commands.append(cmd)

	def execute(self, timeout=None, encoding='utf-8'):
		# command to run blender in the background as a python console
		cmd = ['blender', '--background', '--python-console'] + self._args
		# setup a *.blend file to load when running blender
		if self._blend_file is not None:
			cmd.insert(1, self._blend_file)
		# compile the source code from the py_file and the stacked commands
		code = self._code + ''.join(l + '\n' for l in self._commands)
		byte_code = bytearray(code, encoding)
		# run blender and the compiled source code
		p = Popen(cmd, stdin=PIPE, stdout=PIPE, stderr=PIPE)
		out, err = p.communicate(byte_code, timeout=timeout)
		# get the output and print it
		out = out.decode(encoding)
		err = err.decode(encoding)
		skip = len(self._code.splitlines())
		Blender._print(out, err, skip, self._commands)
		# empty the commands list
		self._commands = []

	def _register(self, func):
		call = lambda *a, **k: self(func, *a, **k)
		setattr(self, func, call)

	def _print(out, err, skip, commands):
		def replace(out, lst):
			lst = [''] * lst if isinstance(lst, Number) else lst
			for string in lst:
				i, j = out.find('>>> '), out.find('... ')
				ind = max(i, j) if min(i, j) == -1 else min(i, j)
				out = out[:ind] + string + out[ind + 4:]
			return out
		out = replace(out, skip)
		out = replace(out, ['${ ' + c + ' }$\n' for c in commands])
		print('${Running on Blender}$')
		out = out[:-19]
		print(out)
		err = err[err.find('(InteractiveConsole)') + 22:]
		if err:
			print(err)
		print('${Blender Done}$')
		return out, err



if __name__ == '__main__':
	base_path = "D:\\mywork\\sublime\\vgd\\frames"
	for i in range(20):
		b = Blender('init.py','trils.blend')
		# b.color_object(obj_name="Cube",colors=(0.05,0.8,0.05))
		my_random_vector = [np.random.random(),np.random.random(),np.random.random(),
			np.random.randint(7),np.random.randint(7),np.random.randint(-180,180)]
		b.basic_experiment(obj_name="Cube", vec=my_random_vector)
		b.save_image(128,128,path=base_path,name=str(i))
		# b.save_file()
		b.execute()