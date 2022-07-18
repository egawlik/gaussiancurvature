from __future__ import division
from dolfin import *
import numpy
from petsc4py import PETSc
import sys,argparse
from ufl import indices
from scipy.special.orthogonal import p_roots

# Use the same random numbers every time so that results are reproducible
numpy.random.seed(0)

# Parse the command line arguments
total = len(sys.argv)
cmdargs = str(sys.argv)
print ("The total numbers of args passed to the script: %d " % total)
print ("Args list: %s " % cmdargs)
print ("Script name: %s" % str(sys.argv[0]))

parser = argparse.ArgumentParser()
parser.add_argument('-r', '--num_refinements', default=3, type=int) # mesh refinement level
parser.add_argument('-o', '--order', default=0, type=int) # order of Regge finite elements
parser.add_argument('-v', '--orderV', default=1, type=int) # order of Lagrange finite elements
parser.add_argument('-w', '--orderW', default=1, type=int) # order of Nedelec finite elements
parser.add_argument('-c', '--convergence_test', default=0, type=int) # set equal to 1 to just compute the L2 error and quit
args = parser.parse_args()
num_refinements = args.num_refinements
order = args.order
orderV = args.orderV
orderW = args.orderW
convergence_test = args.convergence_test

# Create mesh
nx = 2**(num_refinements)
ny = nx
mesh = RectangleMesh(Point(-1.0, -1.0), Point(1.0, 1.0), nx, ny, "left")
deltax = 2.0 / nx
deltay = 2.0 / ny

# Define Dirichlet boundary (x = -1 or x = 1 or y = -1 or y = 1) 
def boundary(x):
    return x[0] < -1.0 + DOLFIN_EPS or x[0] > 1.0 - DOLFIN_EPS or x[1] < -1.0 + DOLFIN_EPS or x[1] > 1.0 - DOLFIN_EPS

# Perturb vertices so we don't get fooled by superconvergence phenomena
x = mesh.coordinates()[:,0]
y = mesh.coordinates()[:,1]
xtilde = x.copy()
ytilde = y.copy()
for i in range(len(x)):
    if not boundary(mesh.coordinates()[i,:]):
        xtilde[i] += 0.2*deltax*(2*numpy.random.rand()-1)
        ytilde[i] += 0.2*deltay*(2*numpy.random.rand()-1)
xytilde = numpy.array([xtilde, ytilde]).transpose()
#mesh.coordinates()[:] = xytilde

# Define finite element spaces
Sigma = FunctionSpace(mesh, "Regge", order)
V = FunctionSpace(mesh, "CG", orderV)
W = FunctionSpace(mesh, "N1curl", orderW)

# Quadrature degree
parameters["form_compiler"]["quadrature_degree"] = 12

# Define boundary condition
bcV = DirichletBC(V, Constant(0.0), boundary)
bcW = DirichletBC(W, Constant((0.0,0.0)), boundary)

# Define trial functions, test functions, and data
kappa = TrialFunction(V)
connection = TrialFunction(W)
v = TestFunction(V)
alpha = TestFunction(W)
delta = Constant(((1.0,0.0),(0.0,1.0)))
#fx = Expression(" 1.0*(x[0]-1.0/3.0*x[0]*x[0]*x[0])", degree=3)
#fy = Expression(" 1.0*(x[1]-1.0/3.0*x[1]*x[1]*x[1])", degree=3)
#kappaexact = Expression("81*(1-x[0]*x[0])*(1-x[1]*x[1]) / pow(9 + x[0]*x[0]*pow(x[0]*x[0]-3,2) + x[1]*x[1]*pow(x[1]*x[1]-3,2),2)", degree=12)
#gexact = Expression( ( ("1.0+fx*fx", "fx*fy"), ("fx*fy", "1.0+fy*fy") ), fx=fx, fy=fy, degree = 6, domain=mesh)
#fx = Expression(" 1.0*(x[0]-1.0/3.0*x[0]*x[0]*x[0])", degree=12)
#fy = Expression(" 1.0*(x[1]-1.0/3.0*x[1]*x[1]*x[1])", degree=12)
#kappaexact = Expression("81*(1-x[0]*x[0])*(1-x[1]*x[1]) / pow(9 + x[0]*x[0]*pow(x[0]*x[0]-3,2) + x[1]*x[1]*pow(x[1]*x[1]-3,2),2)", degree=12)
#gexact = Expression( ( ("1.0+fx*fx", "fx*fy"), ("fx*fy", "1.0+fy*fy") ), fx=fx, fy=fy, degree = 12, domain=mesh) 
fx = Expression("-0.5*pi*sin(0.5*pi*x[0])", degree=12)
fy = Expression("-0.5*pi*sin(0.5*pi*x[1])", degree=12)
kappaexact = Expression("4*pow(pi,4)*cos(0.5*pi*x[0])*cos(0.5*pi*x[1]) / pow(-2*(4+pi*pi) + pi*pi*(cos(pi*x[0])+cos(pi*x[1])),2)", degree=12)
gexact = Expression( ( ("1.0+fx*fx", "fx*fy"), ("fx*fy", "1.0+fy*fy") ), fx=fx, fy=fy, degree=12, domain=mesh)
g = interpolate(gexact,Sigma)
#g = project(gexact,Sigma)
#g = gexact
sigma = g-delta

# Print the interpolation error for g
print("L2norm(g - gexact) = ", errornorm(gexact,g,degree_rise=6))

# Normal and tangent vectors
n = FacetNormal(mesh)
rotmat = as_matrix( [[0,1],[-1,0]] )
tau = -rotmat*n

# Christoffel symbols of the second kind associated with a metric g
def christoffel(g):
    i,j,k,l = indices(4)
    gamma = as_tensor(0.5 * inv(g)[k,l] * ( g[l,i].dx(j) + g[l,j].dx(i) - g[i,j].dx(l) ), (k,i,j))
    return gamma

# Riemannian Hessian of a scalar field v
def hess(v,gamma):
    i,j,k = indices(3)
    hessv = as_tensor(v.dx(i).dx(j) - gamma[k,i,j]*v.dx(k), (i,j))
    #hessv = gradoneform(grad(v),gamma) # equivalent
    return hessv

# Covariant derivative of a one-form alpha
def gradoneform(alpha,gamma):
    i,j,k = indices(3)
    gradalpha = as_tensor(alpha[i].dx(j) - gamma[k,i,j]*alpha[k], (i,j))
    return gradalpha

# Gaussian curvature of g
def curvature(g):
    i,j,k,l = indices(4)
    gamma = christoffel(g)
    gausscurv = 0.5 * inv(g)[i,j] * ( gamma[k,i,j].dx(k) - gamma[k,i,k].dx(j) + gamma[l,i,j]*gamma[k,k,l] - gamma[l,i,k]*gamma[k,j,l] )
    return gausscurv

# Calculate the integral of b( (1-t)*delta+t*g, g-delta, v ) from t=0 to t=1
[nodes,weights] = p_roots(20)
nodes = (nodes+1)/2
weights = weights/2
rhs = 0.0
rhsconn = 0.0
for iq in range(len(weights)):
    w = weights[iq]
    t = nodes[iq]
    G = (1-t)*delta + t*g

    gamma = christoffel(G)
    invG = inv(G)
    sqrtdetG = sqrt(det(G))
    ll = sqrt(dot(tau,G*tau))
    tauG = tau / ll
    nG = invG*n*sqrtdetG / ll
    SGsigma = sigma - G*tr(invG*sigma)

    tauGp = tauG('+') # equal to -tauG('-')
    nGp = nG('+') # NOT equal to -nG('-')
    llp = ll('+') # equal to ll('-')
    sigmap = sigma('+')

    rhs = rhs + w * dot(tauGp, sigmap*tauGp) * jump( grad(v), nG ) * llp * dS
    rhs = rhs + w * dot(tauG , sigma *tauG ) *  dot( grad(v), nG ) * ll  * ds
    rhs = rhs + w * tr( invG*SGsigma*invG*hess(v,gamma) ) * sqrtdetG * dx

    rhsconn = rhsconn + w * dot(tauGp, sigmap*tauGp) * jump( alpha, nG ) * llp * dS
    rhsconn = rhsconn + w * dot(tauG , sigma *tauG ) *  dot( alpha, nG ) * ll  * ds
    rhsconn = rhsconn + w * tr( invG*SGsigma*invG*gradoneform(alpha,gamma) ) * sqrtdetG * dx

rhs = 0.5*rhs
rhsconn = 0.5*rhsconn
lhs = kappa*v*sqrt(det(g))*dx
#lhsconn = dot(connection,alpha)*sqrt(det(g))*dx
lhsconn = dot(connection,inv(g)*alpha)*sqrt(det(g))*dx

# Compute solution
kappa = Function(V)
connection = Function(W)
solve(lhs == rhs, kappa, bcV)
solve(lhsconn == rhsconn, connection, bcW)

# Check that the exterior coderivative of the connection equals the curvature
diff = assemble( kappa*v*sqrt(det(g))*dx - dot(connection,inv(g)*grad(v))*sqrt(det(g))*dx )
bcV.apply(diff)
print("vectornorm(d*connection-kappa) = ", norm(diff))

# Compare with the exact Gaussian curvature
print("L2norm(kappa  - kappaexact) = ", errornorm(kappaexact,kappa,degree_rise=6))
if convergence_test:
    sys.exit()

# Since the connection we computed is really approximating the Hodge star
# of the connection one-form, compute the Hodge star for plotting purposes
RT = FunctionSpace(mesh, "RT", orderW)
bcRT = DirichletBC(RT, Constant((0.0,0.0)), boundary)
starconnection = project(sqrt(det(g))*rotmat*inv(g)*connection, RT, bcRT)

# Save solution
file = File("results/kappa.pvd")
file << kappa
file = File("results/connection.pvd")
file << starconnection

#kappa3 = interpolate(kappaexact,V)
#vert2dof = V.dofmap().entity_dofs(mesh, 0)
#for i in range(mesh.num_vertices()):
#    print(kappa.vector()[vert2dof[i]])
#print(sqrt(assemble(kappaexact*kappaexact*sqrt(det(g))*dx)))

#########################################################################################################
#########################################################################################################
#########################################################################################################
# Now let's check if we get the same result by computing angle defects, jumps in geodesic curvature, etc.

# Calculate the interior angles of every triangle with respect to the metric g.
# The entry i,j of "angle" will contain the interior angle of triangle i at vertex j,
# where j is the global index of vertex j.  Most of the entries of "angle" will be 0.
# In particular, if j is not a vertex of triangle i, then angle[i][j]=0.
angle = numpy.zeros((mesh.num_cells(),mesh.num_vertices()))
for cell in cells(mesh):
    # Get the edges of the current triangle
    edges = facets(cell)

    # We need to walk through pairs of edges and compute the angle between them.
    # Since "edges" is an iterator, it doesn't seem straightforward to access pairs of edges.
    # I will brute force it. (Probably there is a better way.)
    # First let's store the first edge in the list.
    k = 0
    for e in facets(cell):
        if k==0:
            e0 = e
            break
        k += 1

    # Now let's walk through the pairs of edges.
    e2 = e0
    stop = False
    while not stop:
        e1 = e2 # First edge in the pair
        e2 = next(edges,-1) # Second edge in the pair
        if e2==-1: # If we're on the last edge, loop back around.
            stop = True
            e2 = e0
        # Identify the vertex that's shared by e1 and e2.
        for v1 in vertices(e1):
            for v2 in vertices(e2):
                if v1.index() == v2.index():
                    # Normal vectors to e1 and e2.
                    n1 = e1.normal()
                    n2 = e2.normal()
                    # These might not be outward pointing normal vectors.  To see which way
                    # the normal vector n1 points, let's look at how the 2 triangles
                    # adjacent to the edge e1 are ordered.  The first adjacent triangle is 
                    # e1.entities(2)[0] and the second one (if any) is e1.entities(2)[1].
                    if not (e1.entities(2)[0] == cell.index()):
                        n1 *= -1
                    # Similarly for e2.
                    if not (e2.entities(2)[0] == cell.index()):
                        n2 *= -1

                    # Evaluate the Regge metric g at the vertex v1.
                    coord = [v1.point().x(),v1.point().y()]
                    #coord[0] -= 100*DOLFIN_EPS * (n1[0] + n2[0])
                    #coord[1] -= 100*DOLFIN_EPS * (n1[1] + n2[1])
                    gval = numpy.empty(4, dtype=float)
                    g.eval_cell(gval,coord,cell)

                    # Compute the angle between the two tangent vectors with respect to g.
                    tau1 = numpy.zeros(2)
                    tau2 = numpy.zeros(2)
                    tau1[0] = -n1[1]
                    tau1[1] =  n1[0]
                    tau2[0] = -n2[1]
                    tau2[1] =  n2[0]
                    tau1dottau2 = ( tau1[0]*gval[0]*tau2[0] + tau1[0]*gval[1]*tau2[1] + tau1[1]*gval[2]*tau2[0] + tau1[1]*gval[3]*tau2[1] )
                    ll1 =     sqrt( tau1[0]*gval[0]*tau1[0] + tau1[0]*gval[1]*tau1[1] + tau1[1]*gval[2]*tau1[0] + tau1[1]*gval[3]*tau1[1] )
                    ll2 =     sqrt( tau2[0]*gval[0]*tau2[0] + tau2[0]*gval[1]*tau2[1] + tau2[1]*gval[2]*tau2[0] + tau2[1]*gval[3]*tau2[1] )
                    tau1dottau2 = tau1dottau2/ll1/ll2

                    theta = numpy.arccos( numpy.clip( -tau1dottau2, -1, 1) )
                    angle[cell.index()][v1.index()] = theta


# Compute angle defects by summing around each vertex and subtracting from 2*pi
angledefect = numpy.zeros(mesh.num_vertices())
for j in range(mesh.num_vertices()):
    if not boundary(mesh.coordinates()[j,:]):
        angledefect[j] = 2*pi
        for i in range(mesh.num_cells()):
            if angle[i][j]>0.0:
                angledefect[j] -= angle[i][j]

# Initialize the right-hand side of the linear system to zero.
rhs2 = 2.0*DOLFIN_EPS*v*dx # Seems impossible to assemble a zero rhs, so make it tiny instead

# If we're working with a Regge metric of order > 0, we need to compute the 
# jumps in geodesic curvature across each edge, as well as the curvature inside
# each triangle.
if order>0:
    ll = sqrt(dot(tau,g*tau))
    taug = tau / ll
    ng = inv(g)*n*sqrt(det(g)) / ll
    gamma = christoffel(g)
    i,j,k = indices(3)
    gradtautau = as_vector( -0.5 * taug[j] * taug[i] * dot(taug,g.dx(j)*taug) + gamma[i,j,k]*taug[j]*taug[k], (i) )
    rhs2 -= jump(ng,g*gradtautau) * v('+') * ll('+') * dS
    rhs2 += curvature(g) * v * sqrt(det(g)) * dx

# Assemble the mass matrix and the right-hand side
M, b = assemble_system(lhs, rhs2, bcV)

# Add angle defects to the right-hand side
vert2dof = V.dofmap().entity_dofs(mesh, 0)
for i in range(mesh.num_vertices()):
    pt = mesh.coordinates()[i,:]
    b[vert2dof[i]] += angledefect[i]

# Compute solution
kappa2 = Function(V)
solve(M, kappa2.vector(), b)

# Compare with the exact Gaussian curvature
print("L2norm(kappa2 - kappaexact) = ", errornorm(kappaexact,kappa2,degree_rise=6))

# Save solution
file = File("results/kappatwo.pvd")
file << kappa2

# Check that the two kappa's are the same
#print("vectornorm(kappa - kappa2) = ", norm(kappa.vector() - kappa2.vector()) )
print("norm(kappa - kappa2) = ", sqrt(assemble((kappa-kappa2)*(kappa-kappa2)*sqrt(det(g))*dx)) )
