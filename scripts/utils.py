import xml.etree.ElementTree as ET
import os

def get_episodes(ep_path: str) -> list[int]:
    """Get the episodes data

    Returns:
        sorted_episodes (list[int]): the sorted episodes data
    Raises:
        FileNotFoundError: If the episodes folder does not exist
    """

    eps = list()
    if os.path.exists(ep_path):
        for file in os.listdir(ep_path):
            episode = int(file.split("ep")[1].split(".csv")[0])
            eps.append(episode)
    else:
        raise FileNotFoundError(f"Episodes folder does not exist!")


    return sorted(eps)


def clear_SUMO_files(sumo_path, ep_path, remove_additional_files=False):
    '''
        Clear SUMO files that are empty or not in the episodes folder.
        Works only for the consecutive files with the same name.
        The files are named as <file_name>_<episode>.xml

        This is a destructive function, it will remove files from the directory!
    '''
    file_id = 1
    episode = 1

    file_name = "detailed_sumo_stats"
    
    while True:
        # check if file exists
        file_path = os.path.join(sumo_path, f"{file_name}_{episode}.xml")
        if os.path.exists(file_path):
            # read xml file and check if <tripinfos> is empty (no <tripinfo> elements)
            try:
                tree = ET.parse(file_path)
            except ET.ParseError:
                print(f"Error parsing XML file: {file_path}")
                break
            root = tree.getroot()
            if len(root.findall("tripinfo")) == 0:
                # remove the file
                os.remove(file_path)
                # print(f"Removed empty file: {file_path}")
            else:
                # rename to the next file_id
                new_file_path = os.path.join(sumo_path, f"{file_name}_{file_id}.xml")
                os.rename(file_path, new_file_path)
                # print(f"Renamed file {file_path} to {new_file_path}")
                file_id += 1
        else:
            break
        episode += 1

    file_id = 1
    episode = 1

    file_name = "sumo_stats"

    while True:
        # check if file exists
        file_path = os.path.join(sumo_path, f"{file_name}_{episode}.xml")
        if os.path.exists(file_path):
            # read xml file and check if <vehicle loaded=0>
            try:
                tree = ET.parse(file_path)
            except ET.ParseError:
                print(f"Error parsing XML file: {file_path}")
                break
            root = tree.getroot()
            vehicle = root.find("vehicles")
            if vehicle is not None and vehicle.attrib.get("loaded") == "0":
                # remove the file
                os.remove(file_path)
                # print(f"Removed empty file: {file_path}")
            else:
                # rename to the next file_id
                new_file_path = os.path.join(sumo_path, f"{file_name}_{file_id}.xml")
                os.rename(file_path, new_file_path)
                # print(f"Renamed file {file_path} to {new_file_path}")
                file_id += 1
        else:
            break
        episode += 1
    if remove_additional_files:
        episodes = get_episodes(ep_path)
        # remove SUMO files that are not in the episodes
        for file in os.listdir(sumo_path):
            if file.endswith(".xml"):
                episode = int(file.split("_")[-1].split(".")[0])
                if episode not in episodes:
                    os.remove(os.path.join(sumo_path, file))
                    # print(f"Removed file: {file}")
