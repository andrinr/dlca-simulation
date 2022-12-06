import taichi as ti

ti.init(arch=ti.gpu)

N = 8
dt = 5e-5
dx = 1 / N
rho = 4e1
NF = 2 * N**2  # number of faces
NV = (N + 1)**2  # number of vertices
E, nu = 4e3, 0.2  # Young's modulus and Poisson's ratio
mu, lam = E / 2 / (1 + nu), E * nu / (1 + nu) / (1 - 2 * nu)  # Lame parameters
#ball_pos, ball_radius = ti.Vector([0.5, 0.0]), 0.31
bar_pos = ti.Vector([0.2, 0.2])
damping = 14.5

gravity = ti.Vector.field(2, float, ())
attractor_pos = ti.Vector.field(2, float, ())
attractor_strength = ti.field(float, ())


@ti.data_oriented
class SoftBodyRect:
    def __init__(self):
        self.pos = ti.Vector.field(2, float, NV, needs_grad=True)
        self.vel = ti.Vector.field(2, float, NV)
        self.f2v = ti.Vector.field(3, int, NF)
        self.B = ti.Matrix.field(2, 2, float, NF)
        self.F = ti.Matrix.field(2, 2, float, NF, needs_grad=True)
        self.V = ti.field(float, NF)
        self.phi = ti.field(float, NF)
        self.U = ti.field(float, (), needs_grad=True)  # total potential energy

    @ti.kernel
    def initialize(self, scale : float, offset : ti.math.vec2):
        for i, j in ti.ndrange(N, N):
            k = (i * N + j) * 2
            a = i * (N + 1) + j
            b = a + 1
            c = a + N + 2
            d = a + N + 1
            self.f2v[k + 0] = [a, b, c]
            self.f2v[k + 1] = [c, d, a]

        for i, j in ti.ndrange(N + 1, N + 1):
            k = i * (N + 1) + j
            self.pos[k] = ti.Vector([i, j]) / N * scale + offset
            self.vel[k] = ti.Vector([0, 0])

        for i in range(NF):
            ia, ib, ic = self.f2v[i]
            a, b, c = self.pos[ia], self.pos[ib], self.pos[ic]
            B_i_inv = ti.Matrix.cols([a - c, b - c])
            self.B[i] = B_i_inv.inverse()

    @ti.kernel
    def advance(self):
        for i in range(NV):
            acc = -self.pos.grad[i] / (rho * dx**2)
            g = gravity[None] * 0.8 + attractor_strength[None] * (
                attractor_pos[None] - self.pos[i]).normalized(1e-5)
            self.vel[i] += dt * (acc + g * 40)
            self.vel[i] *= ti.exp(-dt * damping)
        for i in range(NV):
            # ball boundary condition:
            #disp = pos[i] - ball_pos
            #disp2 = disp.norm_sqr()
            #if disp2 <= ball_radius**2:
            #    NoV = vel[i].dot(disp)
            #    if NoV < 0:
            #        vel[i] -= NoV * disp / disp2
            cond = (self.pos[i][1] < bar_pos[0]) & (self.vel[i] < 0)
            # rect boundary condition:
            for j in ti.static(range(self.pos.n)):
                if cond[j]:
                    self.vel[i][j] = 0
            self.pos[i] += dt * self.vel[i]

    @ti.kernel
    def update_U(self):
        for i in range(NF):
            ia, ib, ic = self.f2v[i]
            a, b, c = self.pos[ia], self.pos[ib], self.pos[ic]
            self.V[i] = abs((a - c).cross(b - c))
            D_i = ti.Matrix.cols([a - c, b - c])
            self.F[i] = D_i @ self.B[i]

        for i in range(NF):
            F_i = self.F[i]
            log_J_i = ti.log(F_i.determinant())
            phi_i = mu / 2 * ((F_i.transpose() @ F_i).trace() - 2)
            phi_i -= mu * log_J_i
            phi_i += lam / 2 * log_J_i**2
            self.phi[i] = phi_i
            self.U[None] += self.V[i] * phi_i

    def paint_phi(self, gui):
        pos_ = self.pos.to_numpy()
        phi_ = self.phi.to_numpy()
        f2v_ = self.f2v.to_numpy()
        a, b, c = pos_[f2v_[:, 0]], pos_[f2v_[:, 1]], pos_[f2v_[:, 2]]
        k = phi_ * (10 / E)
        gb = (1 - k) * 0.5
        #gui.triangles(a, b, c, color=ti.rgb_to_hex([k + gb, gb, gb]))
        gui.triangles(a, b, c, color=0xff0000)


def main():
    mesh1 = SoftBodyRect()
    mesh2 = SoftBodyRect()

    mesh1.initialize(0.25, ti.Vector([0.1, 0.6]))
    mesh2.initialize(0.2, ti.Vector([0.6, 0.3]))
   
    gravity[None] = [0, -1]

    gui = ti.GUI('FEM128')
    print(
        "[Hint] Use WSAD/arrow keys to control gravity. Use left/right mouse buttons to attract/repel. Press R to reset."
    )
    for frame in range(100):
        for i in range(200):
            with ti.ad.Tape(loss=mesh1.U):
                mesh1.update_U()
            
            mesh1.advance()
            
            with ti.ad.Tape(loss=mesh2.U):
                mesh2.update_U()

            mesh2.advance()

        mesh1.paint_phi(gui)
        mesh2.paint_phi(gui)

        #gui.circle(mouse_pos, radius=15, color=0x336699)
        #gui.circle(ball_pos, radius=ball_radius * 512, color=0x666666)
        #gui.circles(pos.to_numpy(), radius=2, color=0xffaa33)
        gui.triangle([0, bar_pos.y], [1.0, bar_pos.y], [1.0, 0.0], color=0x00ff00)
        gui.triangle([0, bar_pos.y], [0.0, 0.0], [1.0, 0.0], color=0x00ff00)
        #gui.show()

        filename = f'export2/frame_{frame:05d}.png'   # create filename with suffix png
        print(f'Frame {frame} is recorded in {filename}')
        gui.show(filename)  # export and show in GUI

if __name__ == '__main__':
    main()