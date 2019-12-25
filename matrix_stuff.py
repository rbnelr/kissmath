
def indent(txt, level):
  return ''.join('    '*level + l for l in txt.splitlines(1))

def mat_cell_letters(size):
	return [[chr(ord('a') + r*size + c) for c in range(size)] for r in range(size)]

def define_letterify(T, size):
	define_txt = ['#define LETTERIFY']

	cell_letter = mat_cell_letters(size)
	
	for r in range(size):
		for c in range(size):
			define_txt.append(f'{T} {cell_letter[r][c]} = mat.arr[{r}][{c}];')

	return ' \\\n'.join(define_txt)

def letterify(size):
	cell_letter = mat_cell_letters(size)
	
	return lambda r,c: cell_letter[r][c]

def optimize(T, txt): # optimize m2 - m4 to not do redudant calculations
	def _optimize(T, txt, op, var_len):
		import re

		def multiple_replace(string, rep_dict):
			pattern = re.compile("|".join([re.escape(k) for k in sorted(rep_dict,key=len,reverse=True)]), flags=re.DOTALL)
			return pattern.sub(lambda x: rep_dict[x.group(0)], string)

		ops = [tuple(multiple_replace(x, {' ':'', '(':'', ')':''}).split(op)) for x in \
			re.findall(rf"(?:\b|\(?)[a-z]{{{var_len}}}\s*\{op}\s*[a-z]{{{var_len}}}(?:\)?|\b)", txt)]
		ops_txt = ''

		ops_set = set(ops)
		ops = [ (lm, ops.count(lm)) for lm in ops_set ]

		for lm,count in ops:
			if count > 1:
				a,b = lm
				shortform = f'{a}{b}'
				txt = re.sub(rf"\b{a}\s*\{op}\s*{b}\b", shortform, txt)
				ops_txt += f'{T} {shortform} = {a} {op} {b};\n'
		return ops_txt, txt

	ops_txt = ''

	res_ops_txt, txt = _optimize(T, txt, '*', 1)
	ops_txt += res_ops_txt

	res_ops_txt, txt = _optimize(T, txt, '-', 2)
	ops_txt += ('\n' if ops_txt else '') + res_ops_txt

	return ops_txt +'\n'+ txt if ops_txt else txt
	
def stats(txt):
	import re
	muls = txt.count("*")
	adds = txt.count("+") + txt.count("-")
	divs = len(re.findall(r"[^/]/[^/]", txt))
	stats = f'// {muls} muls, {adds} adds, {divs} divs = {muls + adds + divs} ops'
	return stats

def gen_determinate_code(T, size):
	cell = letterify(size)

	txt, expr = _gen_determinate_code(T, size, cell)

	unopt_stats = stats(txt + expr)
	txt = optimize(T, txt)
	opt_stats = stats(txt + expr)
	
	txt = 'LETTERIFY\n\n' + txt
	txt = txt + f'return %s;' % expr
	
	txt = f'// optimized from:  {unopt_stats}\n// to:              {opt_stats}\n' + txt
	return txt

# get cell r,c coords in matrix for minor matrix at minor_r,minor_c
def get_minor_cell(r,c, minor_r,minor_c):
	return r+1 if r >= minor_r else r, c+1 if c >= minor_c else c

def sign(r,c): # signs of cofactors
	return ('+','-')[ (r%2) ^ (c%2) ]

def _gen_determinate_code(T, size, cell, det_chain='det', depth=0):
	txt = ''
	
	if size == 1:
		expr = f'{cell(0,0)}'

	elif size == 2:
		expr = f'{cell(0,0)}*{cell(1,1)} - {cell(0,1)}*{cell(1,0)}'

	else:
		expr = []

		for c in range(size):
			det = f'{det_chain}_{cell(0,c)}'

			minor_txt, minor_expr = _gen_determinate_code(T, size-1,
				lambda minor_r,minor_c: cell(*get_minor_cell(minor_r,minor_c, 0,c)), det, depth+1)

			txt += indent(minor_txt, depth)

			expr += [sign(0,c) + f'{cell(0,c)}*({minor_expr})']
		if txt:
			txt += '\n'

		expr = ('\n' if size >= 4 else ' ').join(expr)

	return txt, expr

def gen_inverse_code(M, T, size):
	cell = letterify(size)

	txt = ''
	txt += f'{T} det;\n{{ // clac determinate\n'

	det_txt, det_expr = _gen_determinate_code(T, size, cell)
	txt += det_txt
	txt += f'det = {det_expr};\n}}\n'
	txt += f'{T} inv_det = {T}(1) / det;\n'
	txt += f'{T} ninv_det = -inv_det;\n\n'

	txt += f'// calc cofactor matrix\n\n'
	
	for r in range(size):
		for c in range(size):
			cofac = f'cofac_{r}{c}'

			det_txt, det_expr = _gen_determinate_code(T, size-1,
				lambda minor_r,minor_c: cell(*get_minor_cell(minor_r,minor_c, r,c)), cofac)

			txt += indent(det_txt, 1) + f'{T} {cofac} = {det_expr};\n' + ('\n' if det_txt else '')


	txt += f'\n{M} ret;\n\n'
	
	for r in range(size):
		for c in range(size):
			transp_r, transp_c = c, r
			
			txt += f'ret.arr[{r}][{c}] = cofac_{transp_r}{transp_c} * {"n" if sign(r,c) == "-" else " "}inv_det;\n'

	txt += f'\nreturn ret;'
	
	unopt_stats = stats(txt)
	txt = optimize(T, txt)
	opt_stats = stats(txt)
	
	txt = 'LETTERIFY\n\n'+ txt

	txt = f'// optimized from:  {unopt_stats}\n// to:              {opt_stats}\n' + txt
	return txt