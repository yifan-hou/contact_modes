import numpy as np
from sympy import lambdify, nsimplify, pi, pprint, simplify, symbols, trigsimp
from sympy.matrices import Matrix

from contact_modes.symbolic import *

from .shape import Shape
from .link import Link

DEBUG = True

class Chain(Shape):
    def __init__(self, n, r=0.5, l=1.0):
        self.n = n
        self.r = r
        self.l = l
        self.link_mesh = Link(r, l)
        # self.link_mesh = None
        self.links = [ChainLink(self.link_mesh) for i in range(n)]

    def generate(self):
        self.generate_forward_kinematics()
        self.generate_jacobian()

    def generate_forward_kinematics(self):
        # Get link parameters.
        n_links = self.n
        n_dofs = 2 * (self.n - 1)
        link_radius = self.r
        link_length = self.l

        # Create transforms gₛₗ(0) from base frame to each link l.
        g_sl0 = [None] * n_links
        g_sl0[0] = eye_rat(4)
        for i in range(1, n_links):
            g_sl = eye_rat(4)
            g_sl[0,3] = Rational(i * (link_length + 2 * link_radius))
            if i % 2 == 1:
                g_sl[0:3,0:3] = exp3(Matrix(3,1,[1,0,0]), pi/2)
            g_sl0[i] = g_sl
            if DEBUG:
                print('g_sl0[i]')
                pprint(g_sl0[i])

        # Create revolute joint twists for each DOF, 2 per link.
        joint_twists = []
        w1 = Matrix(3,1,[0,0,1])
        w2 = Matrix(3,1,[0,-1,0])
        for i in range(1, n_links):
            q1 = Matrix(3,1,[g_sl0[i-1][0,3] + link_length/2, 0, 0])
            x1 = zeros(6,1)
            x1[0:3,0] = -w1.cross(q1)
            x1[3:6,0] =  w1
            for j in range(6):
                x1[j,0] = Rational(x1[j,0])
            x2 = zeros(6,1)
            q2 = Matrix(3,1,[g_sl0[i][0,3] - link_length/2, 0, 0])
            x2[0:3,0] = -w2.cross(q2)
            x2[3:6,0] =  w2
            for j in range(6):
                x2[j,0] = Rational(x2[j,0])
            joint_twists.append([x1, x2])
            if DEBUG:
                print('x1')
                pprint(x1)
                print('x2')
                pprint(x2)

        # Create symbolic DOFs.
        q = []
        k = 0
        for i in range(1, n_links):
            q1 = symbols('q%d' % k)
            k += 1
            q2 = symbols('q%d' % k)
            k += 1
            q.append([q1, q2])

        # Create transforms gₛₗ(θ) from base frame to each link l.
        g_sl = [None] * n_links
        g_sl[0] = eye_rat(4)
        exps = [None] * n_links
        exps[0] = eye_rat(4)
        for i in range(1, n_links):
            q1 = q[i-1][0]
            q2 = q[i-1][1]
            x1 = joint_twists[i-1][0]
            x2 = joint_twists[i-1][1]

            exp_x1 = exp6(x1, q1)
            exp_x2 = exp6(x2, q2)

            if DEBUG:
                print('exp x1')
                pprint(exp_x1)
                print('exp x2')
                pprint(exp_x2)

            # exps[i] = exps[i-1] * exp_x1 * exp_x2
            exps[i] = exps[i-1] * trigsimp(exp_x1 * exp_x2)
            # exps[i] = trigsimp(exps[i])
            # g_sl[i] = trigsimp(exps[i] * g_sl0[i])
            # g_sl[i] = exps[i] * g_sl0[i]
            g_sl[i] = exps[i-1] * trigsimp(exp_x1 * exp_x2 * g_sl0[i])
            if DEBUG:
                print('g_sl[%d]' % i)
                pprint(g_sl[i])

        # Assign transforms to each link.
        dofs = []
        for q1, q2 in q:
            dofs.append(q1)
            dofs.append(q2)
        for i in range(n_links):
            self.links[i].set_tf_expr(lambdify(dofs, g_sl[i]))
    
    def generate_jacobian(self):
        # We will computer the body jacobian using the formula
        #   Jₛₜᵇ(θ) = Ad⁻¹(gₛₜ) [(∂gₛₜ/∂θ₁⋅gₛₜ⁻¹)ᵛ ⋯ (∂gₛₜ/∂θₙ⋅gₛₜ⁻¹)ᵛ]
        # But we will compute gₛₜ and Dgₛₜ symbolically
        #   Dgₛₜ = [∂gₛₜ/∂θ₁ ⋯ ∂gₛₜ/∂θₙ]
        # and perform the rest numerically.
        Dg = []

        # Convert to body jacobian.
    
    def num_dofs(self):
        return 2*self.n-2

    def set_state(self, q):
        for l in self.links:
            l.set_state(q)

    def get_state(self, q):
        pass

    def draw(self, shader):
        for link in self.links:
            link.draw(shader)

class ChainLink(Shape):
    def __init__(self, mesh):
        super(ChainLink, self).__init__()
        self.mesh = mesh

    def set_tf_expr(self, g_sl):
        self.g_sl = g_sl

    def set_state(self, q):
        self.get_tf_world().set_matrix(self.g_sl(*q))

    def draw(self, shader):
        self.mesh.set_tf_world(self.get_tf_world())
        self.mesh.draw(shader)