import numpy as np
import h5py as hf
import scipy.spatial as sp
from tqdm import tqdm


class GaussianKernel:
    """
    Class for taking measurements in the observation layer
    """
    def __init__(self,
                 load_directory: str,
                 number_kernel: int,
                 number_agents: int = None,
                 colloid_pos: np.ndarray = None,
                 colloid_velocity: np.ndarray = None,
                 predator_pos: np.ndarray = None,
                 kernel_pos: np.ndarray = None,
                 kernel_width: np.ndarray = None
                 ):
        """

        Parameters
        ----------
        load_directory: str
            Input directory of the measurement
        number_kernel: int
            Number of kernels that measure position and velocity of
            agents
        number_agents: int
            Number of used agents
        colloid_pos: np.ndarray (timesteps, number_agents, 2)
            array of the colloid position at each timestep (n_time_steps default = 1000)
        colloid_pos: np.ndarray (timesteps, number_agents, 2)
            array of the colloid velocities at each timestep
        predator_pos: np.ndarray
            array of predator position at each timestep
        kernel_pos: np.ndarray (number_kernel, 2)
            array of kernel positions
        kernel_width: np.ndarray (number_kernel, 1)
            array of the width of the kernels
        """
        self.load_directory = load_directory
        self.number_kernel = number_kernel
        self.number_agents = number_agents
        self.colloid_pos = colloid_pos
        self.colloid_velocity = colloid_velocity
        self.predator_pos = predator_pos
        self.kernel_pos = kernel_pos
        self.kernel_width = kernel_width

    def load_data(self):
        """
        Reads in the data out of the load_directory
        Position data is agent and predator positions at each time step
        excludes predator position out of each timestep

        Returns
        -------
        Sets value for already initialized variables
        colloid_pos: np.ndarray
            (n_time_steps, number_agent, dimension)
        colloid_velocity: np.ndarray
        predator_pos: np.ndarray
        number_agents: int
        """
        with hf.File(f"{self.load_directory}/trajectory.hdf5", 'r') as db:
            position_data = db["colloids"]["Unwrapped_Positions"][:]
            velocity_data = db["colloids"]["Velocities"][:]
            # type_data = db["colloids"]["Types"][:]
        colloid_pos_temp, colloid_velocity_temp = [], []
        predator_pos_temp = []
        for i in range(len(position_data)):
            colloid_pos_temp.append(position_data[i][:-1])
            colloid_velocity_temp.append(velocity_data[i][:-1])
            predator_pos_temp.append(position_data[i][-1])

        self.colloid_pos = np.array(colloid_pos_temp)
        self.colloid_velocity = np.array(colloid_velocity_temp)
        self.predator_pos = np.array([pos[:-1] for pos in predator_pos_temp])
        self.number_agents = len(self.colloid_pos[0])

    def set_kernels(self):
        """
        Chooses Kernel positions by choosing a random timestep
            + random agent for each Kernel
        Chooses Kernel widths by setting it to distance to 5th neighbour using scipy.spatial.KDTree

        Returns
        -------
        Sets value for already initialized variables
        kernel_pos: np.ndarray
            (number_kernel, dimension)
        kernel_width: np.ndarray
            (number_kernel)
        """
        random_timesteps = np.random.randint(0, len(self.colloid_pos), size=self.number_kernel)
        random_agents = np.random.randint(0, self.number_agents, size=self.number_kernel)
        self.kernel_pos = self.colloid_pos[random_timesteps, random_agents]

        self.kernel_width = []
        for i in range(len(self.kernel_pos)):
            points = self.colloid_pos[random_timesteps[i]]
            kdtree = sp.KDTree(points)
            distances, _ = kdtree.query(self.kernel_pos[i], k=6)
            self.kernel_width.append(np.max(distances))

        self.kernel_width = np.array(self.kernel_width)

    @staticmethod
    def psi(colloid_pos_at_t, kernel_pos, kernel_width):
        """
        Kernel distribution for all kernels at each timestep

        Parameters
        ----------
        colloid_pos_at_t: np.ndarray
            (number_agents, dimension)
        kernel_pos: np.ndarray
            (number_kernel, dimension)
        kernel_width: np.ndarray
            (number_kernel)

        Returns
        -------
        psi_value: list
            Distribution value for each colloid to each kernel
            (number_agents)
        """
        return np.exp(-np.dot((colloid_pos_at_t - kernel_pos), (colloid_pos_at_t - kernel_pos)) / (2*kernel_width))

    def calc(self, colloid_pos_at_t, colloid_velocity_at_t, kernel_pos, kernel_width):
        """
        Calculates r vector components r1,r2,r3 by summing over the Psi function of each kernel for all agents
        but only at one timestep
        r1 position, r2 x-velocity, r3 y-velocity

        Parameters
        ----------
        colloid_pos_at_t: np.ndarray
            (number_agents, dimension)
        colloid_velocity_at_t: np.ndarray
            (number_agents, dimension)
        kernel_pos: np.ndarray
            (number_kernel, dimension)
        kernel_width: np.ndarray
            (number_kernel)
        Returns
        -------
        r_vector_at_t: np.ndarray
            the system describing vector at one timestep
            (3 * number_kernel)
        """
        r1 = []
        r2, r3 = [], []
        for i in range(len(kernel_pos)):
            r1.append(0)
            r2.append(0)
            r3.append(0)
            for j in range(len(colloid_pos_at_t)):
                psi_temp = self.psi(colloid_pos_at_t[j], kernel_pos[i], kernel_width[i])
                r1[i] += psi_temp
                r2[i] += psi_temp * colloid_velocity_at_t[j][0]
                r3[i] += psi_temp * colloid_velocity_at_t[j][1]

        # r1 = [np.sum(self.psi(xt,ci,wi[0])) for (ci,wi) in list(zip(c,w))]
        # something wrong with input dimensions, prob bc of tuple
        return np.concatenate([r1, r2, r3])

    def run(self):
        """

        Returns
        -------
        data_set : dict
            Data set for training a model:
            {"inputs": (n_time_steps, dimension), "targets": (n_steps, 2)}

        """
        self.load_data()
        self.set_kernels()
        r = []
        for xt, vt in tqdm(zip(self.colloid_pos, self.colloid_velocity)):
            r.append(self.calc(xt, vt, self.kernel_pos, self.kernel_width))
        return {"inputs": np.array(r), "targets": self.predator_pos}
