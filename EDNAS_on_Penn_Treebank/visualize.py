import sys
import genotypes
from graphviz import Digraph


def plot(genotype, filename):
  g = Digraph(
      format='pdf',
      edge_attr=dict(fontsize='20', fontname="times"),
      node_attr=dict(style='filled', shape='rect', align='center', fontsize='20', height='0.5', width='0.5', penwidth='2', fontname="times"),
      engine='dot')
  g.body.extend(['rankdir=LR'])

  g.node("x[t]", fillcolor='whitesmoke')
  g.node("h[t-1]", fillcolor='whitesmoke')
  g.node("0", fillcolor='yellow1')
  g.edge("x[t]", "0", fillcolor="gray")
  g.edge("h[t-1]", "0", fillcolor="gray")
  steps = len(genotype)

  for i in range(1, steps + 1):
    g.node(str(i), fillcolor='yellow1')

  for i, (op, j) in enumerate(genotype):
    g.edge(str(j), str(i + 1), label=op, fillcolor="gray")

  g.node("h[t]", fillcolor='lawngreen')
  for i in range(1, steps + 1):
    g.edge(str(i), "h[t]", fillcolor="gray")

  g.render(filename, view=True)


if __name__ == '__main__':
  if len(sys.argv) != 2:
    print("usage:\n python {} ARCH_NAME".format(sys.argv[0]))
    sys.exit(1)

  genotype_name = sys.argv[1]
  try:
    genotype = eval('genotypes.{}'.format(genotype_name))
  except AttributeError:
    print("{} is not specified in genotypes.py".format(genotype_name)) 
    sys.exit(1)

  plot(genotype.recurrent, "recurrent")

