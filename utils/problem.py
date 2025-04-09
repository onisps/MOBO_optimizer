import datetime
import open3d as o3d
import trimesh
import pathlib
from utils.global_variable import *
from utils.compute_utils import get_history_output as get_history_output
from utils.compute_utils import run_abaqus
from utils.fea_results_utils import process_fea_data
from utils.project_utils import purgeFiles
from utils.create_geometry_utils import trimesh_to_shell
from utils.create_geometry_utils import generate_leaflet_pointcloud
from utils.create_input_files import write_inp_shell, write_inp_contact
import os

now = str(datetime.datetime.now()).replace(' ', '_').replace(':', '-').split('.')[0]
now = now[:-3]

pathToAbaqus = str(pathlib.Path(__file__).parent.resolve()) + '/abaqusWF/'
path_utils = str(pathlib.Path(__file__).parent.resolve())

# procedure parameters class
class LeafletProblem:
    """
    Class wrapper for parametrized simulation and analysis of leaflet geometries using Abaqus FEA.

    This class encapsulates procedures for automated generation, meshing, boundary condition application,
    material definition, and numerical simulation of aortic or mitral valve leaflets.
    It supports both single-leaflet and contact-based multi-leaflet configurations.

    Attributes:
        baseName (str): Base string identifier for file naming and procedural tracking.
        cpus (int): Number of CPU threads used for Abaqus simulations (default: 4).
        logging (bool): Flag to enable logging of procedural messages to `.log` file.
        mesh_step (float): Characteristic mesh step size for leaflet surface discretization.

    Methods:
        run_procedure(params) -> dict:
            Executes the leaflet modeling workflow based on the selected problem type
            (either 'leaflet_single' or 'leaflet_contact') and returns simulation objectives.
    """

    baseName = None
    cpus = 4
    logging = True
    mesh_step = None

    def __init__(self, cpus, logging, baseName, mesh_step):
        self.baseName = baseName
        self.cpus = cpus
        self.logging = logging
        self.mesh_step = mesh_step

    # ==================================================================================================================
    # __________________________________________RUN PROCEDURE___________________________________________________________
    # ==================================================================================================================

    def run_procedure(self, params) -> dict:
        """
        Executes the entire simulation pipeline (geometry → meshing → FEA → parsing → metrics)
        depending on the problem type defined in global settings (`get_problem_name()`).

        Args:
            params (tuple): Parametric inputs for leaflet geometry generation:
                - Single-leaflet: (Lstr, ANG, CVT, LAS) or (HGT, Lstr, THK, ANG, CVT, LAS)
                - Contact-leaflet: Same structure, but internally generates 3 rotated parts

        Returns:
            dict: A dictionary with keys `"objectives"` and `"constraints"` containing
                  computed simulation metrics (e.g., LMN_open, Smax, VMS, Helicopter effect).

        Raises:
            Exceptions during geometry creation, FEA execution, or ODB parsing are caught and logged.
        """

        def run_leaflet_single(self, params) -> dict:
            try:
                baseName = self.baseName + '_' + now
                try:
                    HGT, Lstr, THK, ANG, CVT, LAS = params
                except:
                    Lstr, ANG, CVT, LAS = params
                    HGT = 11
                    THK = 0.3
                DIA = 22.98
                Lift = 0
                SEC = 120
                EM = 3.2
                mesh_step = self.mesh_step
                tangent_behavior = 0.05
                normal_behavior = 0.2
                try:
                    pointsInner, _, _, _, pointsHullLower, _, points, _, finalRad, currRad, message = \
                        generate_leaflet_pointcloud(HGT=HGT, Lstr=Lstr, SEC=SEC, DIA=DIA, THK=0.5,
                                                    ANG=ANG, Lift=Lift, CVT=CVT, LAS=LAS, mesh_step=mesh_step)
                except Exception as e:
                    raise e

                k = 1.1
                flag_calk_k = True
                while flag_calk_k:
                    try:
                        o3d.utility.set_verbosity_level(o3d.utility.VerbosityLevel(0))
                        pcd = o3d.geometry.PointCloud()
                        pcd.points = o3d.utility.Vector3dVector(points.T)
                        mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_alpha_shape(pcd, alpha=0.5 * k)
                        _ = o3d.io.write_triangle_mesh('./utils/geoms/temp.ply', mesh, write_vertex_normals=True)

                        mesh = trimesh.load_mesh('./utils/geoms/temp.ply')
                        mesh.fix_normals()  # fix wrong normals
                        mesh.export('./utils/geoms/temp.stl')
                        flag_calk_k = False
                    except:
                        k += 0.1
                try:
                    shellNode, shellEle, fixed_bc = trimesh_to_shell(points=mesh.vertices, elements=mesh.faces,
                                                                     pointsInner=pointsInner,
                                                                     pointsHullLower=pointsHullLower, meshStep=mesh_step)
                    message = 'done'
                except Exception as e:
                    raise e

                del mesh
                inpFileName = str(inpDir) + str(baseName)
                jobName = str(baseName) + '_Job'
                modelName = str(baseName) + '_Model'
                partName = str(baseName) + '_Part'
                if get_direction().lower() == 'direct':
                    try:
                        write_inp_shell(
                            fileName=inpFileName + '.inp', Nodes=shellNode, Elements=shellEle,
                            BCfix=fixed_bc, THC=THK, Emod=EM, Dens=9e-10, JobName=jobName,
                            ModelName=modelName, partName=partName, MaterialName='PVA', PressType='vent',
                            press_overclosure='linear', tangent_behavior=tangent_behavior,
                            normal_behavior=normal_behavior
                        )
                        message = run_abaqus(pathToAbaqus, jobName, inpFileName, self.cpus)
                    except Exception as e:
                        raise e

                    # парсим odb, считываем поля, считаем максимумы и площадь открытия, пишем в outFileName
                    try:
                        get_history_output(pathName=pathToAbaqus, odbFileName=jobName + '.odb', cpus=self.cpus)
                    except:
                        raise 'Odb parse problem'

                    try:
                        endPath = pathToAbaqus + 'results/'
                        LMN_op, LMN_cl, Smax, VMS, perf_index, _ = process_fea_data(
                            pathToAbaqus=pathToAbaqus, endPath=endPath, partName=partName,
                            Slim=get_s_lim(), DIA=DIA, SEC=SEC,
                            mesh_step=mesh_step
                        )
                        purgeFiles(endPath, partName, pathToAbaqus, jobName)
                    except Exception as e:
                        raise e

                    if LMN_op < 0:
                        change_direction()
                        try:
                            write_inp_shell(
                                fileName=inpFileName + '.inp', Nodes=shellNode, Elements=shellEle,
                                BCfix=fixed_bc, THC=THK, Emod=EM, Dens=9e-10, JobName=jobName,
                                ModelName=modelName, partName=partName, MaterialName='PVA', PressType='vent',
                                press_overclosure='linear', tangent_behavior=tangent_behavior,
                                normal_behavior=normal_behavior
                            )
                            message = run_abaqus(pathToAbaqus, jobName, inpFileName, self.cpus)
                        except Exception as e:
                            raise e

                        # парсим odb, считываем поля, считаем максимумы и площадь открытия, пишем в outFileName
                        try:
                            get_history_output(pathName=pathToAbaqus, odbFileName=jobName + '.odb', cpus=self.cpus)
                        except:
                            raise 'Odb parse problem'

                        try:
                            endPath = pathToAbaqus + 'results/'
                            LMN_op, LMN_cl, Smax, VMS, perf_index, _ = process_fea_data(
                                pathToAbaqus=pathToAbaqus, endPath=endPath, partName=partName, Slim=get_s_lim(),
                                DIA=DIA, SEC=SEC, mesh_step=mesh_step
                            )
                            purgeFiles(endPath, partName, pathToAbaqus, jobName)
                        except Exception as e:
                            raise e
                        change_direction()
                del fixed_bc, partName, jobName, endPath, modelName, inpFileName

                if LMN_cl < 0:
                    out_lmn_cl = 0
                else:
                    out_lmn_cl = LMN_cl

                objectives_dict = {
                    '1 - LMN_open': 1 - LMN_op,
                    'LMN_open': LMN_op,
                    "LMN_closed": out_lmn_cl,
                    "Smax": Smax
                }

                constraints_dict = {
                    "VMS-Smax": VMS - get_s_lim()
                }


                return {"objectives": objectives_dict, "constraints": constraints_dict}
            except Exception as exept:

                objectives_dict = {
                    'LMN_open': 0.0,
                    '1 - LMN_open': 1.0,
                    "LMN_closed": 1.0,
                    "Smax": 5
                }

                constraints_dict = {
                    "VMS-Smax": 5
                }

                return {"objectives": objectives_dict, "constraints": constraints_dict}

        def run_leaflet_contact(self, params) -> dict:
            try:
                baseName = self.baseName + '_' + now
                try:
                    HGT, Lstr, THK, ANG, CVT, LAS = params
                except:
                    Lstr, ANG, CVT, LAS = params
                    HGT = 11
                    THK = 0.3
                DIA = get_DIA()
                Lift = get_Lift()
                EM = get_EM()
                mesh_step = self.mesh_step
                tangent_behavior = get_tangent_behavior()
                normal_behavior = get_normal_behavior()
                SEC = get_SEC()
                Dens = get_density()
                MaterialName = get_material_name()
                PressType = get_valve_position()
                try:
                    pointsInner, _, _, _, pointsHullLower, _, points, _, finalRad, currRad, message = \
                        generate_leaflet_pointcloud(HGT=HGT, Lstr=Lstr, SEC=SEC, DIA=DIA, THK=0.35,
                                                    ANG=ANG, Lift=Lift, CVT=CVT, LAS=LAS, mesh_step=mesh_step)
                except Exception as e:
                    raise e

                k = 1.1
                flag_calk_k = True
                while flag_calk_k:
                    try:
                        o3d.utility.set_verbosity_level(o3d.utility.VerbosityLevel(0))
                        pcd = o3d.geometry.PointCloud()
                        pcd.points = o3d.utility.Vector3dVector(points.T)
                        mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_alpha_shape(pcd, alpha=0.5 * k)
                        _ = o3d.io.write_triangle_mesh(path_utils+'/geoms/temp.ply', mesh, write_vertex_normals=True)
                        mesh = trimesh.load_mesh(path_utils + '/geoms/temp.ply')
                        mesh.fix_normals()  # fix wrong normals
                        mesh.export(path_utils + '/geoms/temp.stl')
                        flag_calk_k = False
                    except:
                        k += 0.1
                try:
                    shellNode, shellEle, fixed_bc = trimesh_to_shell(points=mesh.vertices, elements=mesh.faces,
                                                                     pointsInner=pointsInner,
                                                                     pointsHullLower=pointsHullLower, meshStep=mesh_step)
                except Exception as e:
                    raise e
                del mesh
                inpFileName = str(inpDir) + str(baseName)
                jobName = str(baseName) + '_Job'
                modelName = str(baseName) + '_Model'
                partName = str(baseName) + '_Part'

                try:
                    write_inp_contact(
                        fileName=inpFileName + '.inp', Nodes=shellNode, Elements=shellEle,
                        BCfix=fixed_bc, THC=THK, Emod=EM, Dens=Dens, JobName=jobName,
                        ModelName=modelName, partName=partName, MaterialName=MaterialName, PressType=PressType,
                        press_overclosure='linear', tangent_behavior=tangent_behavior,
                        normal_behavior=normal_behavior
                    )
                    run_abaqus(pathToAbaqus, jobName, inpFileName, self.cpus)
                except Exception as e:
                    raise e
                try:
                    get_history_output(pathName=pathToAbaqus, odbFileName=jobName + '.odb')
                except Exception as e:
                    print(f'[problem.run_leaflet_contact] ODB parse problem! > {e}')
                    raise 'Odb parse problem'

                try:
                    endPath = pathToAbaqus + 'results/'
                    LMN_op, LMN_cl, Smax, VMS, perf_index, heli = process_fea_data(
                        pathToAbaqus=pathToAbaqus, endPath=endPath, partName=partName, Slim=get_s_lim(),
                        DIA=DIA, SEC=SEC, mesh_step=mesh_step
                    )
                    purgeFiles(endPath, partName, pathToAbaqus, jobName)
                except Exception as e:
                    print(f'Change diraction! Direct parce problem: {e}!')
                    change_direction()
                    try:
                        write_inp_contact(
                            fileName=inpFileName + '.inp', Nodes=shellNode, Elements=shellEle,
                            BCfix=fixed_bc, THC=THK, Emod=EM, Dens=Dens, JobName=jobName,
                            ModelName=modelName, partName=partName, MaterialName=MaterialName, PressType=PressType,
                            press_overclosure='linear', tangent_behavior=tangent_behavior,
                            normal_behavior=normal_behavior
                        )
                        run_abaqus(pathToAbaqus, jobName, inpFileName, self.cpus)
                    except Exception as e:
                        raise e
                    try: # already in try: and switched direction
                        get_history_output(pathName=pathToAbaqus, odbFileName=jobName + '.odb')
                    except:
                        raise 'Odb parse problem'

                    try:
                        endPath = pathToAbaqus + 'results/'
                        LMN_op, LMN_cl, Smax, VMS, perf_index, heli = process_fea_data(
                            pathToAbaqus=pathToAbaqus, endPath=endPath, partName=partName, Slim=get_s_lim(),
                            DIA=DIA, SEC=SEC, mesh_step=mesh_step
                        )
                        purgeFiles(endPath, partName, pathToAbaqus, jobName)
                    except:
                        raise e
                    raise e
                del fixed_bc, partName, jobName, endPath, modelName, inpFileName
                # maximization problem
                # maximize LMN_open
                # minimize LMN_cl => maximize -LMN_cl
                # minimize Slim - Smax => maximize Smax - Slim
                # minimize heli => maximize -heli
                objectives_dict = {
                    'LMN_open': LMN_op,
                    "-LMN_closed": -LMN_cl,
                    "Smax - Slim": Smax - get_s_lim(),
                    '-HELI': -heli
                }
                return {"objectives": objectives_dict}
            except Exception as exept:

                objectives_dict = {
                    'LMN_open': -1.0,
                    "-LMN_closed": -1.0,
                    "Smax - Slim": -get_s_lim(),
                    '-HELI': -3
                }

                return {"objectives": objectives_dict}

        problem_name = get_problem_name().lower()
        if problem_name == 'leaflet_single':
            res = run_leaflet_single(self, params)
        elif problem_name == 'leaflet_contact':
            res = run_leaflet_contact(self, params)
        return res


# ====================================================================================================================
# ___________________________________________INIT PROCEDURE___________________________________________________________
# ====================================================================================================================

# абсолютный путь к папке ипутов /inps/, расположенной в папке проекта
inpDir = str(pathlib.Path(__file__).parent.resolve()) + '/inps/'

def init_procedure(param_array):
    """
    Initializes the FEA simulation environment for the selected leaflet or beam model.

    Based on the global problem name (`get_problem_name()`), this function configures:
        - Working directory structure
        - Logging preferences
        - Number of CPUs
        - Mesh resolution
        - Runtime parameters

    It acts as a wrapper for problem-specific setup routines, instantiating and returning
    a `LeafletProblem` (or other designated) object.

    Args:
        param_array (list or tuple): Parametric definition of the geometry. The specific contents
            depend on the selected problem type, e.g., (HGT, Lstr, THK, ANG, CVT, LAS) for leaflet cases.

    Returns:
        LeafletProblem: An instance of the corresponding problem handler class, configured and ready
            to execute the simulation pipeline via `run_procedure()`.

    Raises:
        Exception: If an unsupported problem name is provided.

    Notes:
        - Directory setup includes: `utils/inps`, `utils/logs`, and `utils/geoms`.
        - Logging is configured to store simulation traces in timestamped subfolders.
    """
    def init_procedure_leaf_single(cpus=10, logging=True, baseName='test', mesh_step=0.5):
        # prepare folders for xlsx, inps, logs, geoms
        folders = ['inps', 'logs', 'geoms']
        for folder in folders:
            if not os.path.exists('utils/' + folder):
                os.makedirs('utils/' + folder)
        del folders
        base_folder = str(pathlib.Path(__file__).parent.resolve()) + '/logs/'
        folder_path = base_folder
        problem = LeafletProblem(cpus=cpus, logging=logging, baseName=baseName, mesh_step=get_mesh_step())

        return problem

    def init_procedure_leaf_contact(cpus=10, logging=True, baseName='test'):
        # prepare folders for xlsx, inps, logs, geoms
        folders = ['inps', 'logs', 'geoms']
        for folder in folders:
            if not os.path.exists('utils/' + folder):
                os.makedirs('utils/' + folder)
        del folders
        base_folder = str(pathlib.Path(__file__).parent.resolve()) + '/logs/'
        folder_path = base_folder
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)
            print(f"Created folder: {folder_path}")
        problem = LeafletProblem(cpus=cpus, logging=logging, baseName=baseName, mesh_step=get_mesh_step())

        return problem

    problem_name = get_problem_name().lower()
    cpus = get_cpus()
    if problem_name == 'leaflet_single':
        problem = init_procedure_leaf_single(cpus=cpus, logging=True, baseName=get_base_name())
    elif problem_name == 'leaflet_contact':
        problem = init_procedure_leaf_contact(cpus=cpus, logging=True, baseName=get_base_name())
    else:
        raise Exception(f'Wrong problem name: {problem_name}, you entered: {problem_name}')
    return problem
