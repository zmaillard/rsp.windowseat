import logging
import math
import os

from PIL import Image
from torch.utils.data import Dataset

logger = logging.getLogger(__name__)


def _required_side_for_axis(size: int, nmax: int, min_overlap: int) -> int:
    """Smallest tile side T (1D) so that #tiles <= nmax with overlap >= min_overlap."""
    nmax = max(1, int(nmax))
    if nmax == 1:
        return size
    return math.ceil((size + (nmax - 1) * min_overlap) / nmax)


def _starts(size: int, T: int, min_overlap: int):
    """Uniform stepping with stride = T - min_overlap; last tile flush with edge."""
    if size <= T:
        return [0]
    stride = max(1, T - min_overlap)
    xs = list(range(0, size - T + 1, stride))
    last = size - T
    if xs[-1] != last:
        xs.append(last)
    # monotonic dedupe
    out = []
    for v in xs:
        if not out or v > out[-1]:
            out.append(v)
    return out


class TilingDataset(Dataset):
    def __init__(
        self,
        transform_graph,
        input_folder,
        tiling_w=768,
        tiling_h=768,
        processing_resolution=768,
        max_num_tiles_w=4,
        max_num_tiles_h=4,
        min_overlap_w=64,
        min_overlap_h=64,
        use_short_edge_tile=False,
        **kwargs,
    ) -> None:
        super().__init__()
        self.transform_graph = transform_graph
        self.kwargs = kwargs
        self.disp_name = kwargs.get("disp_name", "tiling_dataset")

        img_paths = sorted(
            os.path.join(input_folder, f)
            for f in os.listdir(input_folder)
            if os.path.isfile(os.path.join(input_folder, f))
        )
        logger.info(
            "TilingDataset: found %d image(s) in %s", len(img_paths), input_folder
        )

        self.filenames = []

        Nw, Nh = int(max_num_tiles_w), int(max_num_tiles_h)
        ow, oh = int(min_overlap_w), int(min_overlap_h)

        for i, p in enumerate(img_paths):
            with Image.open(p) as im:
                W, H = im.size

                # Choose preferred tile size for this image
                if use_short_edge_tile:
                    short_edge = min(W, H)
                    short_edge = max(short_edge, processing_resolution)
                    tiling_w_i = short_edge
                    tiling_h_i = short_edge
                else:
                    tiling_w_i = tiling_w
                    tiling_h_i = tiling_h

                # Optional upscaling if image is smaller than desired tile
                if W < tiling_w_i or H < tiling_h_i:
                    min_side = min(W, H)
                    scale_ratio = tiling_w_i / min_side
                    W = round(scale_ratio * W)
                    H = round(scale_ratio * H)

            pref_side = max(int(tiling_w_i), int(tiling_h_i))

            # Feasible square-side interval [T_low, T_high]
            T_low = max(
                _required_side_for_axis(W, Nw, ow),
                _required_side_for_axis(H, Nh, oh),
                ow + 1,
                oh + 1,
            )
            T_high = min(W, H)

            if T_low > T_high:
                msg = (
                    f"Infeasible square constraints for {os.path.basename(p)}: "
                    f"need T >= {T_low}, but max square inside is {T_high}. "
                    f"Relax max_num_tiles_w/h or overlaps, allow non-square tiles, or pad."
                )
                logger.error(msg)
                raise ValueError(msg)
            else:
                T = max(T_low, min(pref_side, T_high))
                Tw = Th = T
                logger.debug(
                    "Image %s (%dx%d): tile_size=%d",
                    os.path.basename(p),
                    W,
                    H,
                    T,
                )

            # Build starts with axis-specific tile sizes
            xs = _starts(W, Tw, ow)
            ys = _starts(H, Th, oh)

            for y0 in ys:
                for x0 in xs:
                    x1, y1 = x0 + Tw, y0 + Th
                    self.filenames.append([str(p), (x0, y0, x1, y1), False])

            if self.filenames:
                self.filenames[-1][-1] = True

        logger.info("TilingDataset: total tiles=%d", len(self.filenames))

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, index):
        sample = {}
        sample["line"] = self.filenames[index]
        sample["idx"] = index
        logger.debug(
            "TilingDataset.__getitem__: index=%d file=%s tile=%s",
            index,
            self.filenames[index][0],
            self.filenames[index][1],
        )
        self.transform_graph(sample)
        return sample
