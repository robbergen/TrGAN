import chainer
import chainer.functions as F
import chainer.links as L












class AdversaryNetwork(chainer.Chain):

    def __init__(self, bottom_width=100, in_channels=1, top_width=4, mid_ch = 64, wscale = 0.01):
        self.bottom_width = bottom_width
        w = chainer.initializers.Uniform(wscale)
        super(AdversaryNetwork, self).__init__()
        with self.init_scope():
            self.l1 = L.Linear(64 * 64 * 32, 64 * 64, initialW=w, nobias=True)
            self.l2 = L.Linear(64 * 64, 1000, initialW=w, nobias=True)
            self.l3 = L.Linear(1000, bottom_width, initialW=w, nobias=True)

    def __call__(self,x):
        h = F.flatten(x)
        h = F.relu(self.l1(h))
        h = F.relu(self.l2(h))
        h = F.relu(self.l3(h))

        return h