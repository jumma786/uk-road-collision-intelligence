"""DfT road casualty statistics code-to-label mappings."""

SEVERITY = {1: "Fatal", 2: "Serious", 3: "Slight"}

DAY_OF_WEEK = {
    1: "Sunday", 2: "Monday", 3: "Tuesday", 4: "Wednesday",
    5: "Thursday", 6: "Friday", 7: "Saturday",
}

LIGHT_CONDITIONS = {
    1: "Daylight", 4: "Darkness_lit", 5: "Darkness_unlit",
    6: "Darkness_no_lighting", 7: "Darkness_unknown",
}

WEATHER_CONDITIONS = {
    1: "Fine_no_wind", 2: "Raining_no_wind", 3: "Snowing_no_wind",
    4: "Fine_wind", 5: "Raining_wind", 6: "Snowing_wind",
    7: "Fog_mist", 8: "Other", 9: "Unknown",
}

ROAD_SURFACE = {
    1: "Dry", 2: "Wet_damp", 3: "Snow", 4: "Frost_ice",
    5: "Flood", 9: "Unknown",
}

ROAD_TYPE = {
    1: "Roundabout", 2: "One_way", 3: "Dual_carriageway",
    6: "Single_carriageway", 7: "Slip_road", 9: "Unknown",
}

ROAD_CLASS = {
    1: "Motorway", 2: "A_M", 3: "A", 4: "B", 5: "C", 6: "Unclassified",
}

URBAN_RURAL = {1: "Urban", 2: "Rural", 3: "Unallocated"}

SPEED_LIMIT_BINS = {
    "low": [20], "medium": [30, 40], "high": [50, 60, 70],
}

JUNCTION_DETAIL = {
    0: "Not_at_junction", 1: "Roundabout", 2: "Mini_roundabout",
    3: "T_or_staggered", 5: "Slip_road", 6: "Crossroads",
    7: "More_than_4_arms", 8: "Private_drive", 9: "Other",
    13: "T_junction", 14: "Roundabout_new", 15: "Mini_roundabout_new",
    16: "Crossroads_new", 17: "More_than_4_new", 18: "Private_drive_new",
    99: "Unknown",
}

JUNCTION_CONTROL = {
    1: "Authorised_person", 2: "Auto_traffic_signal",
    4: "Give_way_uncontrolled", 9: "Unknown",
}

PEDESTRIAN_CROSSING = {
    0: "None_within_50m", 1: "Zebra", 4: "Pelican_puffin_toucan",
    5: "Pedestrian_phase", 7: "Footbridge_subway", 8: "Central_refuge",
    13: "Zebra_new", 14: "Signal_crossing_new", 15: "Signal_with_phase_new",
    16: "Footbridge_new", 17: "Central_refuge_new", 99: "Unknown",
}

POLICE_OFFICER_ATTEND = {
    1: "Yes", 2: "No", 3: "Yes_self_reporting",
}

VEHICLE_TYPE = {
    1: "Pedal_cycle", 2: "Motorcycle_50cc_under", 3: "Motorcycle_125cc_under",
    4: "Motorcycle_125_500cc", 5: "Motorcycle_over_500cc",
    8: "Taxi", 9: "Car", 10: "Minibus", 11: "Bus_coach",
    19: "Van", 20: "HGV_rigid", 21: "HGV_articulated",
    22: "HGV_unknown", 23: "Motorcycle_electric",
    90: "Other_vehicle", 97: "Motorcycle_unknown_cc",
    98: "Goods_unknown_weight", 17: "Agricultural_vehicle",
}

VEHICLE_MANOEUVRE = {
    1: "Reversing", 2: "Parked", 3: "Waiting_to_go_ahead",
    4: "Slowing_stopping", 5: "Moving_off", 6: "U_turn",
    7: "Turning_left", 8: "Waiting_to_turn_left",
    9: "Turning_right", 10: "Waiting_to_turn_right",
    11: "Changing_lane_left", 12: "Changing_lane_right",
    13: "Overtaking_moving", 14: "Overtaking_static",
    15: "Going_ahead_left_bend", 16: "Going_ahead_right_bend",
    17: "Going_ahead_other", 18: "Going_ahead",
    19: "Going_ahead_new", 99: "Unknown",
}

CASUALTY_CLASS = {1: "Driver_rider", 2: "Passenger", 3: "Pedestrian"}

CASUALTY_TYPE = {
    0: "Pedestrian", 1: "Cyclist", 2: "Motorcyclist_50cc_under",
    3: "Motorcyclist_125cc_under", 4: "Motorcyclist_125_500cc",
    5: "Motorcyclist_over_500cc", 8: "Taxi_occupant",
    9: "Car_occupant", 10: "Minibus_occupant",
    11: "Bus_occupant", 16: "Horse_rider",
    19: "Van_occupant", 20: "HGV_occupant",
    21: "HGV_occupant", 22: "HGV_occupant",
    90: "Other_vehicle_occupant", 97: "Motorcyclist_unknown_cc",
    98: "Goods_vehicle_occupant", 23: "Electric_motorcycle_occupant",
}

PROPULSION_CODE = {
    1: "Petrol", 2: "Diesel", 3: "Electric", 4: "Hybrid_electric",
    8: "Gas", 9: "New_fuel_tech", 10: "Fuel_cells",
    11: "Gas_bi_fuel", 12: "Hybrid_plugin",
}

SEX = {1: "Male", 2: "Female", 9: "Unknown"}

TRUNK_ROAD = {1: "Trunk_including_motorway", 2: "Non_trunk"}

SPECIAL_CONDITIONS = {
    0: "None", 1: "Auto_signal_out", 2: "Auto_signal_defective",
    3: "Road_sign_marking_defective", 4: "Roadworks",
    5: "Road_surface_defective", 6: "Oil_or_diesel",
    7: "Mud", 9: "Unknown",
}

IMD_DECILE = {
    1: "Most_deprived_10pct", 2: "10_20pct", 3: "20_30pct",
    4: "30_40pct", 5: "40_50pct", 6: "50_60pct",
    7: "60_70pct", 8: "70_80pct", 9: "80_90pct",
    10: "Least_deprived_10pct",
}

COLLISION_TABLE_MAPPINGS = {
    "collision_severity": SEVERITY,
    "day_of_week": DAY_OF_WEEK,
    "light_conditions": LIGHT_CONDITIONS,
    "weather_conditions": WEATHER_CONDITIONS,
    "road_surface_conditions": ROAD_SURFACE,
    "road_type": ROAD_TYPE,
    "first_road_class": ROAD_CLASS,
    "urban_or_rural_area": URBAN_RURAL,
    "junction_control": JUNCTION_CONTROL,
    "pedestrian_crossing": PEDESTRIAN_CROSSING,
    "did_police_officer_attend_scene_of_accident": POLICE_OFFICER_ATTEND,
    "trunk_road_flag": TRUNK_ROAD,
    "special_conditions_at_site": SPECIAL_CONDITIONS,
}

VEHICLE_TABLE_MAPPINGS = {
    "vehicle_type": VEHICLE_TYPE,
    "vehicle_manoeuvre": VEHICLE_MANOEUVRE,
    "propulsion_code": PROPULSION_CODE,
    "sex_of_driver": SEX,
}

CASUALTY_TABLE_MAPPINGS = {
    "casualty_class": CASUALTY_CLASS,
    "casualty_type": CASUALTY_TYPE,
    "casualty_severity": SEVERITY,
    "sex_of_casualty": SEX,
}
