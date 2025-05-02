from functools import cached_property
from random import choice
from pathlib import Path
from enum import StrEnum

from shift import (
    TransformerPhaseMapperModel,
    TransformerVoltageMapper,
    TransformerVoltageModel,
    parcels_from_location, 
    get_kmeans_clusters, 
    BalancedPhaseMapper, 
    DistributionGraph,
    TransformerTypes, 
    GeoLocation,
    ParcelModel, 
    PRSG,
    add_distribution_graph_to_plot,
    add_voltage_mapper_to_plot,
    add_phase_mapper_to_plot,
    PlotManager,

)

from gdm.distribution import DistributionSystem
from gdm.distribution.enums import Phase
from infrasys.quantities import Voltage
from shapely.geometry import Polygon
from gdm.distribution.components import (
    DistributionVoltageSource,
    DistributionTransformer,
    DistributionTransformer,
    DistributionBranchBase, 
    MatrixImpedanceBranch,
    DistributionLoad,
)
from gdm.distribution.equipment import (
    PhaseVoltageSourceEquipment,
    VoltageSourceEquipment,
    PhaseLoadEquipment,
    LoadEquipment,
)
from gdm.quantities import (
    PositiveApparentPower,
    PositiveVoltage,
    ReactivePower,
    ActivePower, 
    Resistance,
    Reactance,
    Distance,
    Angle,
)

from scipy.spatial import KDTree
from infrasys import System 
from shift import (
    DistributionSystemBuilder,
    EdgeEquipmentMapper, 
    BaseVoltageMapper,
    BasePhaseMapper,
    ParcelModel,
    GeoLocation,
    NodeModel,
)
from loguru import logger
import shift

class SourceSelection(StrEnum):
    RANDOM = "random"
    CENTER = "center"


BASE_SHIFT_PATH =Path(shift.__file__).parent.parent.parent
MODELS_FOLFER =  BASE_SHIFT_PATH/"tests"/"models"

# catalog_sys = DistributionSystem.from_json(MODELS_FOLFER / "p1rhs7_1247.json")

class AreaBasedLoadMapper(EdgeEquipmentMapper):

    def __init__(
        self,
        graph,
        catalog_sys: System,
        voltage_mapper: BaseVoltageMapper,
        phase_mapper: BasePhaseMapper,
        points: list[ParcelModel],
    ):

        self.points = points
        super().__init__(graph, catalog_sys, voltage_mapper, phase_mapper)

    def _get_area_for_node(self, node: NodeModel) -> float:
        """Internal function to return point"""
        tree = KDTree(_get_parcel_points(self.points))
        _, idx = tree.query([GeoLocation(node.location.x, node.location.y)], k=1)
        first_indexes = [el for el in idx]
        nearest_point: ParcelModel = self.points[first_indexes[0]]
        return (
            Polygon(nearest_point.geometry).area if isinstance(nearest_point.geometry, list) else 0
        )

    @cached_property
    def node_asset_equipment_mapping(self):
        node_equipment_dict = {}

        for node in self.graph.get_nodes():
            node_equipment_dict[node.name] = {}
            area = self._get_area_for_node(node)
            if area > 10 and area < 30:
                kw = 1.2
            elif area <= 10:
                kw = 0.8
            else:
                kw = 1.3

            num_phase = len(self.phase_mapper.node_phase_mapping[node.name] - set(Phase.N))
            node_equipment_dict[node.name][DistributionLoad] = LoadEquipment(
                name=f"load_{node.name}",
                phase_loads=[
                    PhaseLoadEquipment(
                        name=f"load_{node.name}_{idx}",
                        real_power=ActivePower(kw / num_phase, "kilowatt"),
                        reactive_power=ReactivePower(0, "kilovar"),
                        z_real=0,
                        i_real=0,
                        p_real=1,
                        z_imag=0,
                        i_imag=0,
                        p_imag=1,
                    )
                    for idx in range(num_phase)
                ],
            )

            if DistributionVoltageSource in node.assets:
                node_equipment_dict[node.name][DistributionVoltageSource] = VoltageSourceEquipment(
                    name="vsource_test",
                    sources=[
                        PhaseVoltageSourceEquipment(
                            name=f"vsource_{idx}",
                            r0=Resistance(1e-5, "ohm"),
                            r1=Resistance(1e-5, "ohm"),
                            x0=Reactance(1e-5, "ohm"),
                            x1=Reactance(1e-5, "ohm"),
                            voltage=Voltage(27, "kilovolt"),
                            angle=Angle(0, "degree"),
                        )
                        for idx in range(3)
                    ],
                )

        return node_equipment_dict


def _get_parcel_points(parcels: list[ParcelModel]) -> list[GeoLocation]:
    return [el.geometry[0] if isinstance(el.geometry, list) else el.geometry for el in parcels]

def get_center_of_parcels(parcels: list[ParcelModel]) -> GeoLocation:
    goemetries =[(geometry.latitude, geometry.longitude) for parcel in parcels for geometry in parcel.geometry]  
    average_location = tuple(sum(t[i] for t in goemetries) / len(goemetries) for i in range(len(goemetries[0])))
    return GeoLocation(latitude=average_location[0], longitude=average_location[1])

def get_random_location(parcels: list[ParcelModel]) -> GeoLocation:
    goemetries =[geometry for parcel in parcels for geometry in parcel.geometry]  
    return choice(goemetries)

def shift_model_builder(
    parcel_source: str | GeoLocation | list[GeoLocation],
    catalog_sys: DistributionSystem,
    radius_in_meters: float = 1000,
    homes_per_distribution_secondary: int = 2,
    source_placement: SourceSelection = SourceSelection.CENTER,
    is_single_phase: bool = True,
    primary_voltage_level_in_kilovolts: float = 7.2,
    secondary_voltage_level_in_kilovolts: float = 0.12,
    gdm_model_name: str = "gdm_model",  

) -> dict:

    
    parcels = parcels_from_location(parcel_source, max_distance=Distance(radius_in_meters, "m"))
    logger.info(f"Found {len(parcels)} parcels")
    num_clusters = int(len(parcels) / homes_per_distribution_secondary)
    clusters = get_kmeans_clusters(max([num_clusters, 1]), _get_parcel_points(parcels))
    logger.info(f"Created {len(clusters)} clusters")
    if source_placement == SourceSelection.CENTER:
        source_location = get_center_of_parcels(parcels)
    else:
        source_location = get_random_location(parcels)
    logger.info(f"Source location: {source_location}")  
    builder = PRSG(groups=clusters, source_location=source_location)
    graph = builder.get_distribution_graph()
    logger.info(f"Created graph with {len(list(graph.get_nodes()))} nodes and {len(list(graph.get_edges()))} edges")  
    new_graph = DistributionGraph()

    for node in graph.get_nodes():
        new_graph.add_node(node)

    for from_node, to_node, edge in graph.get_edges():
        if edge.edge_type == DistributionBranchBase:
            edge.edge_type = MatrixImpedanceBranch
        new_graph.add_edge(from_node, to_node, edge_data=edge)

    graph_plot_manager = PlotManager(center=get_center_of_parcels(parcels))
    add_distribution_graph_to_plot(graph, graph_plot_manager)

    tr_type = TransformerTypes.SPLIT_PHASE if is_single_phase else TransformerTypes.THREE_PHASE 
    mapper = [
        TransformerPhaseMapperModel(
            tr_name=el.name,
            tr_type=tr_type,
            tr_capacity=PositiveApparentPower(
                25,
                "kilova",
            ),
            location=new_graph.get_node(from_node).location,
        )
        for from_node, _, el in new_graph.get_edges()
        if el.edge_type is DistributionTransformer
    ]
    logger.info(f"Created transformer mappers")
    phase_mapper = BalancedPhaseMapper(new_graph, method="greedy", mapper=mapper)
    add_phase_mapper_to_plot(phase_mapper, graph_plot_manager)
    logger.info(f"Created phase mapper")
    voltage_mapper = TransformerVoltageMapper(
        new_graph,
        xfmr_voltage=[
            TransformerVoltageModel(
                name=el.name,
                voltages=[
                    PositiveVoltage(primary_voltage_level_in_kilovolts, "kilovolt"), 
                    PositiveVoltage(secondary_voltage_level_in_kilovolts, "kilovolt")
                ],
            )
            for _, _, el in new_graph.get_edges()
            if el.edge_type is DistributionTransformer
        ],
    )
    add_voltage_mapper_to_plot(voltage_mapper, graph_plot_manager)
    logger.info(f"Created voltage mapper")
    eq_mapper = AreaBasedLoadMapper(
        new_graph,
        catalog_sys=catalog_sys,
        voltage_mapper=voltage_mapper,
        phase_mapper=phase_mapper,
        points=parcels,
    )
    logger.info(f"Created equipment mapper")
    builder = DistributionSystemBuilder(
        name="Test system",
        dist_graph=new_graph,
        phase_mapper=phase_mapper,
        voltage_mapper=voltage_mapper,
        equipment_mapper=eq_mapper,
    )
    logger.info(f"Sucessfully built the gdm system")
    sys = builder.get_system()
    sys.name = gdm_model_name
    graph_plot_manager.show()
    return sys