import swarmrl as srl
from swarmrl.models.interaction_model import Action
import numpy as np
import h5py as hf


class TrajectoryFromFile(srl.models.InteractionModel):
    def __init__(self,
                 load_directory: str,
                 colloid_type: int,
                 particle_gamma_translation: int,
                 particle_gamma_rotation: int,
                 system_runner,
                 ):
        """
        Parameters
        ----------
        load_directory: str
            Input directory of the position data
        colloid_type: int
            Type of colloid that is controlled
        particle_gamma_translation, particle_gamma_rotation: int, int
            Friction coefficient of the colloid, if None it is taken from the system_runner
            (There was no other easy way of getting the friction coefficient, if set manually to a particle type)
        system_runner: SystemRunner
            espressoMD.SystemRunner object
        """
        self.index_tracker = -1
        self.colloid_type = colloid_type
        self.gamma = particle_gamma_translation
        self.gamma_rotation = particle_gamma_rotation
        if self.gamma is None and self.gamma_rotation is None:
            self.gamma, self.gamma_rotation = system_runner.get_friction_coefficients(self.colloid_type)

        db = hf.File(f"{load_directory}/trajectory.hdf5")
        self.colloid_pos = db["Wanted_Positions"][:]
        # there are only positions in the trajectory file not velocities

    def calc_action(self, colloids):
        """
        Calculates the action to get to the next position in the trajectory file
        """
        self.index_tracker += 1
        mass = 1
        actions = []
        for colloid in colloids:
            if not colloid.type == self.colloid_type:
                continue
            pos = self.colloid_pos[self.index_tracker]
            pos1 = self.colloid_pos[self.index_tracker + 1]
            #force = 2/self.t**2 * (pos1 - pos) - 2/self.t * colloid.velocity
            #force = ((pos1-pos) - eps*colloid.velocity)/eps
            force = (pos1 - pos - colloid.velocity * 0.01) * 2*mass/0.01**2
            force_value = np.linalg.norm(force)
            new_direction = force / force_value
            actions.append(Action(force=0.0005*force_value, new_direction=new_direction))
            print("Wanted:", pos, "actual:", colloid.pos, "Force:", force)
        return actions
