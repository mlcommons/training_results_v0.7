import dual_net
from absl import app, flags

flags.DEFINE_string('input_graph', None,
                    'Path of input graph')

flags.DEFINE_string('dst', None,
                    'Path of output file in minigo format')

FLAGS = flags.FLAGS

def main(argv):
    dual_net.convert_pb_to_minigo(FLAGS.input_graph, FLAGS.dst)

if __name__ == '__main__':
    app.run(main)
