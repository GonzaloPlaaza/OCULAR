import torch 
import pandas as pd
from torch.utils.data.dataset import Dataset

class RetinalSegDataset(Dataset):
    def __init__(self, 
                 csv_path: str, 
                 transforms=None,
                 problem_type: str = "multi_class",
                 crossing_is_class: bool = False,
                 dataset_name: str = None,
                 ignore_od: bool = True,
                 no_labels: bool = False,
                 ffa: bool = False):
        
        self.no_labels = no_labels
        self.ffa = ffa
        df = pd.read_csv(csv_path)
        self.build_data(df)
                
        self.transforms = transforms
        self.IGNORE_INDEX = 255
        assert problem_type in ["multi_class", "binary_class", "multi_label"], "problem_type must be either 'multi_class', 'binary_class' or 'multi_label'" 
        self.problem_type = problem_type
        self.crossing_is_class = crossing_is_class  
        self.ignore_od = ignore_od  
        self.dataset_name = dataset_name

        if self.dataset_name is not None:
            #Keep only samples from the specified dataset
            self.data = [d for d in self.data if d["unique_subject_id"].startswith(self.dataset_name)]
            print(f"Dataset {self.dataset_name} specified, keeping {len(self.data)} samples.")

    def build_data(self, df):
        """
        Build self.data from the csv file.
        Args:
            df: pandas dataframe with the data
        """

        fields = {
        "unique_subject_id": "unique_subject_id",
        "image": "image_path"}

        if not self.no_labels:
            fields.update({
                "arteries": "arteries_path",
                "veins": "veins_path",
                "vessels": "vessels_path",
                "major_arteries": "major_arteries_path",
                "major_veins": "major_veins_path",
                "uncertain_vessels": "uncertain_vessels_path",
                "bifurcations_arteries": "bifurcations_arteries_path",
                "bifurcations_veins": "bifurcations_veins_path",
                "crossings": "crossings_path",
                "crossings_roi": "crossings_roi_path",
                "od": "od_path",
                "top": "top",
                "bottom": "bottom",
                "left": "left",
                "right": "right",
            })

        if self.ffa:
            fields.update({
                "ffa_a": "ffa_a_path",
                "ffa_av": "ffa_av_path",
            })

        self.data = [
            {k: getattr(row, v) for k, v in fields.items()}
            for row in df.itertuples(index=False)
        ]

    
    def build_labels(self, d: dict) -> torch.Tensor:
        
        #BINARY VESSEL SEGMENTATION MOD
        if self.problem_type == "binary_class":
            assert d["vessels"].dtype == torch.bool
            vessels = d["vessels"].squeeze(dim=0)  # (H,W)

            labels = torch.zeros(vessels.shape, dtype=torch.uint8)  # (H,W)
            labels[vessels] = 1
        
            #Ignore regions (set last so it wins)
            unc = d["uncertain_vessels"].squeeze(dim=0)
            od = d["od"].squeeze(dim=0)
            ignore = unc | (od if self.ignore_od else torch.zeros_like(od))
            labels[ignore] = self.IGNORE_INDEX

        else:
            
            #Construct label map:
            #Sanity checks (dev only), passing these asserts is needed for later logical operations
            assert d["arteries"].dtype == torch.bool
            assert d["veins"].dtype == torch.bool
            assert d["uncertain_vessels"].dtype == torch.bool
            assert d["od"].dtype == torch.bool

            art, vein, vessels, crossings = d["arteries"].squeeze(dim=0), d["veins"].squeeze(dim=0), \
                d["vessels"].squeeze(dim=0), d["crossings"].squeeze(dim=0)  # (H,W)
            if self.problem_type == "multi_label": #Crossings are not considered here, as they will correspond to pixels both artery and vein.
                labels = torch.zeros((3, art.shape[0], art.shape[1]), dtype=torch.uint8)  #(2,H,W)
                labels[0][art] = 1
                labels[1][vein] = 1
                labels[2][crossings] = 1  
            
            elif self.problem_type == "multi_class":
                if self.crossing_is_class:
                    #When crossings are not a class, they should be ignored
                    labels = torch.zeros(art.shape, dtype=torch.uint8)  # (H,W)
                    labels[art],labels[vein], labels[crossings]  = 1, 2, 3

                    #Ignore regions (set last so it wins)
                    unc = d["uncertain_vessels"].squeeze(dim=0)
                    od = d["od"].squeeze(dim=0)
                    ignore = unc | (od if self.ignore_od else torch.zeros_like(od))
                    labels[ignore] = self.IGNORE_INDEX

                else: 
                    #If crossings are not a class, they should be ignored
                    labels = torch.zeros(art.shape, dtype=torch.uint8)  # (H,W)
                    labels[art],labels[vein]  = 1, 2

                    #Ignore regions (set last so it wins)
                    unc = d["uncertain_vessels"].squeeze(dim=0)
                    od = d["od"].squeeze(dim=0)
                    crossings = d["crossings"].squeeze(dim=0)
                    ignore = unc | (od if self.ignore_od else torch.zeros_like(od)) | crossings
                    labels[ignore] = self.IGNORE_INDEX

        return labels
    
    def build_zones(self, d):
        zone_keys = ("major_arteries", "major_veins", "bifurcations_arteries", "bifurcations_veins", "crossings_roi")
                
        zones = {}
        for k in zone_keys:
            m = d[k]
            m = m.squeeze(dim=0) #(1,H,W) -> (H,W)
            zones[k] = m
        return zones
    

    def build_ignore_mask(self, d: dict) -> torch.Tensor:

        ##IGNORE_MASK = 1 FOR VALID PIXELS, 0 FOR IGNORE
        H, W = d["arteries"].shape[1:]
        if self.problem_type == "multi_label":
            #valid pixels = 1, ignore = 0
            unc = d["uncertain_vessels"].squeeze(0)
            od = d["od"].squeeze(0) if self.ignore_od else torch.zeros_like(unc)
            ignore_mask = (~(unc | od)).long()  #1 = keep, 0 = ignore

            #Check shape is correct; it should be (1, H, W)
            if ignore_mask.ndim == 2:
                ignore_mask = ignore_mask.unsqueeze(0)

        else:
            ignore_mask = None

        return ignore_mask

    def __getitem__(self, index):
        
        d = self.data[index] #here d is list of dicts with paths
        img_path = d["image"] #save before to avoid transforms modifying it
        if self.transforms is not None:
            d = self.transforms(d)

        img = d["image"]
        subject_id = d["unique_subject_id"]
        
        if self.no_labels:
            labels, zones, ignore_mask = torch.ones((1, img.shape[1], img.shape[2]), dtype=torch.bool), \
                torch.ones((1, img.shape[1], img.shape[2]), dtype=torch.bool), torch.ones((1, img.shape[1], img.shape[2]), dtype=torch.bool)   
        
        else:
            labels = self.build_labels(d).long()
            zones = self.build_zones(d)
            ignore_mask = self.build_ignore_mask(d)
            if ignore_mask is None:
                ignore_mask = torch.ones_like(labels, dtype=torch.bool)

        if self.ffa:
            #concatenate ffa_a and ffa_av to the image channels
            ffa_a, ffa_av = d["ffa_a"], d["ffa_av"]
            img = torch.cat([img, ffa_a, ffa_av], dim=0)  #(C+2, H, W)

        return img, labels, zones, subject_id, ignore_mask, img_path
        

    def __len__(self):
        return len(self.data)   