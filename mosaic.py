import sys
import os
import logging
from PIL import Image
from multiprocessing import Process, Queue, cpu_count

# Change these 3 config parameters to suit your needs...
TILE_SIZE      = 50		# height/width of mosaic tiles in pixels
TILE_MATCH_RES = 10		# tile matching resolution (higher values give better fit but require more processing)
ENLARGEMENT    = 8		# the mosaic image will be this many times wider and taller than the original

TILE_BLOCK_SIZE = TILE_SIZE / max(min(TILE_MATCH_RES, TILE_SIZE), 1)
WORKER_COUNT = max(cpu_count() - 1, 1)
OUT_FILE = 'mosaic.jpeg'
EOQ_VALUE = None

logging.basicConfig(level=logging.INFO)


FUDGE_MAP = {
    (200, 5850, 250, 5900): 1,
    (7450, 5050, 7500, 5100): 9,
    (6450, 6050, 6500, 6100): 11,
    (2850, 6100, 2900, 6150): 17,
    (5600, 5650, 5650, 5700): 18,
    (5400, 6150, 5450, 6200): 19,
    (7000, 5400, 7050, 5450): 23,
    (1300, 5100, 1350, 5150): 24,
    (1350, 5550, 1400, 5600): 26,
    (9800, 6000, 9850, 6050): 28,
    (6050, 5950, 6100, 6000): 30,
    (10000, 5700, 10050, 5750): 36,
    (9600, 5700, 9650, 5750): 38,
    (5400, 5450, 5450, 5500): 42,
    (500, 5750, 550, 5800): 43,
    (6250, 5800, 6300, 5850): 44,
    (5550, 5900, 5600, 5950): 46,
    (9350, 5650, 9400, 5700): 51,
    (2650, 5450, 2700, 5500): 57,
    (7700, 5200, 7750, 5250): 58,
    (9200, 5100, 9250, 5150): 63,
    (4650, 5400, 4700, 5450): 65,
    (5650, 5050, 5700, 5100): 66,
    (500, 5350, 550, 5400): 67,
    (900, 6150, 950, 6200): 72,
    (8650, 5600, 8700, 5650): 78,
    (9050, 5900, 9100, 5950): 84,
    (7950, 5400, 8000, 5450): 86,
    (9950, 5700, 10000, 5750): 92,
    (2300, 6150, 2350, 6200): 97,
    (3650, 5500, 3700, 5550): 98,
    (9750, 5600, 9800, 5650): 99,
    (9550, 5350, 9600, 5400): 100,
    (450, 6150, 500, 6200): 102,
    (8200, 6100, 8250, 6150): 103,
    (9650, 5900, 9700, 5950): 108,
    (4200, 5150, 4250, 5200): 110,
    (9650, 6000, 9700, 6050): 114,
    (3650, 5800, 3700, 5850): 119,
    (850, 5100, 900, 5150): 121,
    (5700, 5600, 5750, 5650): 129,
    (4850, 5400, 4900, 5450): 131,
    (2350, 5000, 2400, 5050): 134,
    (2300, 5000, 2350, 5050): 135,
    (9600, 5850, 9650, 5900): 136,
    (8750, 5150, 8800, 5200): 137,
    (3200, 6100, 3250, 6150): 138,
    (10100, 5850, 10150, 5900): 143,
    (7250, 6000, 7300, 6050): 144,
    (3200, 5800, 3250, 5850): 1,
    (0, 6000, 50, 6050): 9,
    (3900, 5100, 3950, 5150): 11,
    (4950, 5850, 5000, 5900): 17,
    (7650, 5950, 7700, 6000): 18,
    (8200, 5050, 8250, 5100): 19,
    (3700, 5050, 3750, 5100): 23,
    (8050, 5600, 8100, 5650): 24,
    (2250, 5550, 2300, 5600): 26,
    (2100, 5450, 2150, 5500): 28,
    (8950, 5000, 9000, 5050): 30,
    (8400, 5650, 8450, 5700): 36,
    (1350, 5100, 1400, 5150): 38,
    (7850, 5550, 7900, 5600): 42,
    (3650, 5900, 3700, 5950): 43,
    (2550, 5800, 2600, 5850): 44,
    (7400, 6000, 7450, 6050): 46,
    (6950, 5400, 7000, 5450): 51,
    (4500, 5100, 4550, 5150): 57,
    (1150, 5150, 1200, 5200): 58,
    (8600, 5650, 8650, 5700): 63,
    (10050, 5550, 10100, 5600): 65,
    (700, 5250, 750, 5300): 66,
    (8450, 5700, 8500, 5750): 67,
    (1500, 5500, 1550, 5550): 72,
    (6950, 5650, 7000, 5700): 78,
    (2750, 5700, 2800, 5750): 84,
    (6500, 5150, 6550, 5200): 86,
    (8200, 6150, 8250, 6200): 92,
    (5850, 5200, 5900, 5250): 97,
    (7800, 5600, 7850, 5650): 98,
    (4800, 5950, 4850, 6000): 99,
    (3350, 5250, 3400, 5300): 100,
    (4450, 5700, 4500, 5750): 102,
    (2600, 5250, 2650, 5300): 103,
    (2700, 5250, 2750, 5300): 108,
    (3900, 5550, 3950, 5600): 110,
    (5200, 5900, 5250, 5950): 114,
    (7700, 5650, 7750, 5700): 119,
    (1050, 5850, 1100, 5900): 121,
    (1950, 5850, 2000, 5900): 129,
    (9400, 5600, 9450, 5650): 131,
    (450, 5900, 500, 5950): 134,
    (5900, 5350, 5950, 5400): 135,
    (4700, 5050, 4750, 5100): 136,
    (1300, 5650, 1350, 5700): 137,
    (3450, 5250, 3500, 5300): 138,
    (4300, 5000, 4350, 5050): 143,
    (8650, 6150, 8700, 6200): 144,
}

class TileProcessor:
    def __init__(self, tiles_directory):
        self.tiles_directory = tiles_directory

    def __process_tile(self, tile_path):
        try:
            img = Image.open(tile_path)
            # tiles must be square, so get the largest square that fits inside the image
            w = img.size[0]
            h = img.size[1]
            min_dimension = min(w, h)
            w_crop = (w - min_dimension) / 2
            h_crop = (h - min_dimension) / 2
            img = img.crop((w_crop, h_crop, w - w_crop, h - h_crop))

            large_tile_img = img.resize((TILE_SIZE, TILE_SIZE), Image.ANTIALIAS)
            small_tile_img = img.resize((int(TILE_SIZE/TILE_BLOCK_SIZE), int(TILE_SIZE/TILE_BLOCK_SIZE)), Image.ANTIALIAS)

            return (large_tile_img.convert('RGB'), small_tile_img.convert('RGB'))
        except:
            return (None, None)

    def get_tiles(self):
        large_tiles = []
        small_tiles = []

        print('Reading tiles from {}...'.format(self.tiles_directory))

        # search the tiles directory recursively
        i = 0
        for root, subFolders, files in os.walk(self.tiles_directory):
           for tile_name in files:
               print('Reading {:40.40}'.format(tile_name), flush=True, end='\r')
               tile_path = os.path.join(root, tile_name)
               large_tile, small_tile = self.__process_tile(tile_path)
               if large_tile:
                   logging.info('tile list: %d %s', i,
                           tile_path)
                   i += 1
                   large_tiles.append(large_tile)
                   small_tiles.append(small_tile)

        print('Processed {} tiles.'.format(len(large_tiles)))

        return (large_tiles, small_tiles)

class TargetImage:
    def __init__(self, image_path):
        self.image_path = image_path

    def get_data(self):
        print('Processing main image...')
        img = Image.open(self.image_path)
        w = img.size[0] * ENLARGEMENT
        h = img.size[1]	* ENLARGEMENT
        large_img = img.resize((w, h), Image.ANTIALIAS)
        w_diff = (w % TILE_SIZE)/2
        h_diff = (h % TILE_SIZE)/2

        # if necessary, crop the image slightly so we use a whole number of tiles horizontally and vertically
        if w_diff or h_diff:
            large_img = large_img.crop((w_diff, h_diff, w - w_diff, h - h_diff))

        small_img = large_img.resize((int(w/TILE_BLOCK_SIZE), int(h/TILE_BLOCK_SIZE)), Image.ANTIALIAS)

        image_data = (large_img.convert('RGB'), small_img.convert('RGB'))

        print('Main image processed.')

        return image_data

class TileFitter:
    def __init__(self, tiles_data):
        self.tiles_data = tiles_data

    def __get_tile_diff(self, t1, t2, bail_out_value):
        diff = 0
        for i in range(len(t1)):
            #diff += (abs(t1[i][0] - t2[i][0]) + abs(t1[i][1] - t2[i][1]) + abs(t1[i][2] - t2[i][2]))
            diff += ((t1[i][0] - t2[i][0])**2 + (t1[i][1] - t2[i][1])**2 + (t1[i][2] - t2[i][2])**2)
            if diff > bail_out_value:
                # we know already that this isn't going to be the best fit, so no point continuing with this tile
                return diff
        return diff

    def get_best_fit_tile(self, img_data):
        best_fit_tile_index = None
        min_diff = sys.maxsize
        tile_index = 0

        # go through each tile in turn looking for the best match for the part of the image represented by 'img_data'
        for tile_data in self.tiles_data:
            diff = self.__get_tile_diff(img_data, tile_data, min_diff)
            if diff < min_diff:
                min_diff = diff
                best_fit_tile_index = tile_index
            tile_index += 1

        return best_fit_tile_index

def fit_tiles(work_queue, result_queue, tiles_data):
    # this function gets run by the worker processes, one on each CPU core
    tile_fitter = TileFitter(tiles_data)

    while True:
        try:
            img_data, img_coords = work_queue.get(True)
            if img_data == EOQ_VALUE:
                break
            tile_index = tile_fitter.get_best_fit_tile(img_data)
            result_queue.put((img_coords, tile_index))
        except KeyboardInterrupt:
            pass

    # let the result handler know that this worker has finished everything
    result_queue.put((EOQ_VALUE, EOQ_VALUE))

class ProgressCounter:
    def __init__(self, total):
        self.total = total
        self.counter = 0

    def update(self):
        self.counter += 1
        print("Progress: {:04.1f}%".format(100 * self.counter / self.total), flush=True, end='\r')

class MosaicImage:
    def __init__(self, original_img):
        self.image = Image.new(original_img.mode, original_img.size)
        self.x_tile_count = int(original_img.size[0] / TILE_SIZE)
        self.y_tile_count = int(original_img.size[1] / TILE_SIZE)
        self.total_tiles  = self.x_tile_count * self.y_tile_count

    def add_tile(self, tile_data, coords):
        img = Image.new('RGB', (TILE_SIZE, TILE_SIZE))
        img.putdata(tile_data)
        self.image.paste(img, coords)

    def save(self, path):
        self.image.save(path)

def build_mosaic(result_queue, all_tile_data_large, original_img_large):
    mosaic = MosaicImage(original_img_large)

    active_workers = WORKER_COUNT
    while True:
        try:
            img_coords, best_fit_tile_index = result_queue.get()

            if img_coords == EOQ_VALUE:
                active_workers -= 1
                if not active_workers:
                    break
            else:
                if img_coords in FUDGE_MAP:
                    best_fit_tile_index = FUDGE_MAP[img_coords]
                logging.info('build_mosaic: %s %d', img_coords,
                        best_fit_tile_index)
                tile_data = all_tile_data_large[best_fit_tile_index]
                mosaic.add_tile(tile_data, img_coords)

        except KeyboardInterrupt:
            pass

    mosaic.save(OUT_FILE)
    print('\nFinished, output is in', OUT_FILE)

def compose(original_img, tiles):
    print('Building mosaic, press Ctrl-C to abort...')
    original_img_large, original_img_small = original_img
    tiles_large, tiles_small = tiles

    mosaic = MosaicImage(original_img_large)

    all_tile_data_large = [list(tile.getdata()) for tile in tiles_large]
    all_tile_data_small = [list(tile.getdata()) for tile in tiles_small]

    work_queue   = Queue(WORKER_COUNT)
    result_queue = Queue()

    try:
        # start the worker processes that will build the mosaic image
        Process(target=build_mosaic, args=(result_queue, all_tile_data_large, original_img_large)).start()

        # start the worker processes that will perform the tile fitting
        for n in range(WORKER_COUNT):
            Process(target=fit_tiles, args=(work_queue, result_queue, all_tile_data_small)).start()

        progress = ProgressCounter(mosaic.x_tile_count * mosaic.y_tile_count)
        for x in range(mosaic.x_tile_count):
            for y in range(mosaic.y_tile_count):
                large_box = (x * TILE_SIZE, y * TILE_SIZE, (x + 1) * TILE_SIZE, (y + 1) * TILE_SIZE)
                small_box = (x * TILE_SIZE/TILE_BLOCK_SIZE, y * TILE_SIZE/TILE_BLOCK_SIZE, (x + 1) * TILE_SIZE/TILE_BLOCK_SIZE, (y + 1) * TILE_SIZE/TILE_BLOCK_SIZE)
                work_queue.put((list(original_img_small.crop(small_box).getdata()), large_box))
                progress.update()

    except KeyboardInterrupt:
        print('\nHalting, saving partial image please wait...')

    finally:
        # put these special values onto the queue to let the workers know they can terminate
        for n in range(WORKER_COUNT):
            work_queue.put((EOQ_VALUE, EOQ_VALUE))

def mosaic(img_path, tiles_path):
    tiles_data = TileProcessor(tiles_path).get_tiles()
    image_data = TargetImage(img_path).get_data()
    compose(image_data, tiles_data)

if __name__ == '__main__':
    if len(sys.argv) < 3:
        print('Usage: {} <image> <tiles directory>\r'.format(sys.argv[0]))
    else:
        mosaic(sys.argv[1], sys.argv[2])

