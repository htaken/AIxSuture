import torch.utils.data as data
import torch
import pandas as pd
import os
import os.path
import numpy as np

from torchvision.transforms import ToTensor, Compose
from transforms import Stack
from PIL import Image
from numpy.random import randint


class DatasetHandler(object):
    def __init__(self, root_path, labels_file_name,
                 num_segments=3, new_length=1, modality='RGB',
                 image_tmpl='img_{:05d}.jpg', transform=None, normalize=None,
                 random_shift=True, test_mode=False, video_suffix="_capture2",
                 return_3D_tensor=False, return_three_channels=False,
                 preload_to_RAM=False, return_trial_id=False, data_seed=42, data_split='70_15_15'):
        """
        params passed on to AachenDataSet
        """

        self.root_path = root_path
        self.labels_file_name = labels_file_name
        self.seed = data_seed
        ar = data_split.split('_')
        self.splits = [int(x)/100 for x in ar]  # converts '70_15_15' -> [0.70, 0.15, 0.15]

        train_data, valid_data, test_data = self._parse_list_files()

        # create AachenDataSet objects to be able to pass to torch DataLoader
        self._train_set = AachenDataSet(root_path, labels_file_name, train_data, num_segments, new_length, modality,
                                        image_tmpl, transform, normalize, random_shift, test_mode,
                                        video_suffix, return_3D_tensor, return_three_channels, preload_to_RAM,
                                        return_trial_id)
        self._valid_set = AachenDataSet(root_path, labels_file_name, valid_data, num_segments*2, # num_segments*2 because training should only see half of video, but validation should be full video
                                        new_length, modality, image_tmpl, transform, normalize, random_shift=False,
                                        test_mode=True,
                                        video_suffix=video_suffix, return_3D_tensor=return_3D_tensor,
                                        return_three_channels=return_three_channels, preload_to_RAM=False,
                                        return_trial_id=True)
        self._test_set = AachenDataSet(root_path, labels_file_name, test_data, num_segments*2, # num_segments*2 because training should only see half of video, but validation should be full video
                                       new_length, modality, image_tmpl, transform, normalize,
                                       random_shift=False, test_mode=True,
                                       video_suffix=video_suffix, return_3D_tensor=return_3D_tensor,
                                       return_three_channels=return_three_channels, preload_to_RAM=False,
                                       return_trial_id=True)

    @property
    def train(self):
        '''
        AachenDataSet object containing the training split
        '''
        return self._train_set

    @property
    def validation(self):
        '''
        AachenDataSet object containing the validation split
        '''
        return self._valid_set

    @property
    def test(self):
        '''
        AachenDataSet object containing the test split
        '''
        return self._test_set

    def _parse_list_files(self):
        """
        Parse file with labels.
        :return: Returns train, validation, and test splits as VideoRecord lists
        """

        # load data
        self.data = pd.read_excel(self.labels_file_name)

        # create a random generator w/ a set seed for reproducibility, use generator to create random splits
        # the splits are generated only on the unique set of the 'STUDENT' column to cluster pre- and post-test
        generator = torch.manual_seed(self.seed)
        filter_student = self.data['STUDENT'].unique()
        # the splits have to be done based on student ID since for each student there are two trials, and both trials
        # should be in the same split
        # splits only of student IDs - have to go back and pick out from actual data list (currently filterA)
        train, validation, test = torch.utils.data.random_split(filter_student, self.splits,
                                                                generator=generator)

        # since train, validation, and test are just lists of indices separating data into the splits, we have to
        # extract the student ID and build the actual split lists based on that
        train_data = self._generate_data_list(train)
        valid_data = self._generate_data_list(validation)
        test_data = self._generate_data_list(test)

        # for record in self.video_list:
        #     frame_count = record.num_frames // self.video_sampling_step
        #     try:
        #         # check whether last frame is there (sometimes gets lost during the extraction process)
        #         self._load_image(os.path.join(self.root_path, record.trial + self.video_suffix), frame_count - 1)
        #     except FileNotFoundError:
        #         frame_count = frame_count - 1
        #     record.frame_count = frame_count
        return train_data, valid_data, test_data

    def _generate_data_list(self, split_list):
        """
        TLDR: rejoins the entered split_list (containing only student IDs) with their respective data (ratings)

        This function takes the train, test, or validation split subset in which only the indices and corresponding
        student IDs are contained. From the subset the student IDs within the split are gathered based on the indices,
        and the corresponding rows of the entire dataset are gathered. The resulting data is then placed into a
        VideoRecord object which are collected in a list.

        :param split_list: train/test/validation split list containing idices for the split and all student IDs (from
                           which the split was generated)
        :return: list of VideoRecord items of the split
        """
        # still here as temp measure in case I want to reuse
        #filterA = self.data[self.data['INVESTIGATOR'].str.contains("A")]  # Filter out only videos rated by rater 'A'

        videorecord_list = []
        for indice in split_list.indices:
            student_ID = split_list.dataset[indice]
            sub_list = self.data.loc[self.data['STUDENT'] == student_ID]
            filter_pre = sub_list[sub_list['TIME'].str.contains("PRE")]
            filter_post = sub_list[sub_list['TIME'].str.contains("POST")]
            averaged_pre = self._average_raters(filter_pre)
            averaged_post = self._average_raters(filter_post)
            averaged_pre_post = pd.concat([averaged_pre, averaged_post])
            sub_videorecord_list = [VideoRecord(row, self.root_path) for index, row in
                                    averaged_pre_post.iterrows() if os.path.exists(os.path.join(self.root_path, row['VIDEO']))]  # pd.dataframe does not automatically iterate over rows, have to tell it to do so
            videorecord_list += sub_videorecord_list

        return videorecord_list

    def _average_raters(self, filtered_list):
        """
        extract single row to reobtain non-numeric values, which row doesn't matter bc all non-numeric values are same accross rows. The row is extracted as a series item, so recast to dataframeand transpose (series is column vector and we need row vector). Replace numeric values with averages.

        :param filtered_list: dataframe filtered for pre and post test results, videos are named differently based on pre and post test
        :return: one row containing averaged OSATS scores
        """
        numeric_cols = self.data.columns[6:]  # columns with OSATS ratings
        averaged_list = filtered_list.mean(numeric_only=True)
        averaged_row = pd.DataFrame(filtered_list.iloc[0]).transpose()  # (not averaged yet just extracting single row)
        averaged_row[numeric_cols] = averaged_list[numeric_cols]  # replace w averages
        return averaged_row


class VideoRecord(object):
    def __init__(self, row, root_folder):
        '''
        Class for video plus labels extracted from a pandas dataframe

        :param row: the row of the dataframe from xlsx file specifying video file name and labels (OSATS scores)
        '''
        self._data = row
        dir = os.path.join(root_folder, self.trial)
        self.frame_count = len([name for name in os.listdir(dir) if os.path.isfile(os.path.join(dir, name))]) # counts all files in dir

    @property
    def trial(self):  # video name
        return self._data['VIDEO']

    # @property
    # def num_frames(self):  # number of frames if sampled at full temporal resolution (30 fps)
    #     return int(self._data[1])

    @property
    def score(self):  # total GRS score (the typo is on purpose, it's like that in the file)
        return int(self._data['GLOBA_RATING_SCORE'])

    @property
    def label(self):
        '''
        Label novice, intermediate, or expert (0, 1, 2).
        Categories chosen based on distribution of average scores
        novice: [8, 16)/8 -> [1, 2) average score
        intermediate: [16, 24)/8 -> [2, 3)
        expert: [24, 41)/8 -> [3, 5]

        :return: novice, intermediate, expert (0, 1, 2), or None if out of range
        '''
        score = round(self.score/8)
        if self.score in range(8, 16):
            return 0
        elif self.score in range(16, 24):
            return 1
        elif self.score in range(24, 41):
            return 2
        else:
            return None


class AachenDataSet(data.Dataset):
    def __init__(self, root_path, labels_file_name, data_list,
                 num_segments=3, new_length=1, modality='RGB',
                 image_tmpl='img_{:05d}.jpg', transform=None, normalize=None,
                 random_shift=True, test_mode=False, video_suffix="_capture2",
                 return_3D_tensor=False, return_three_channels=False,
                 preload_to_RAM=False, return_trial_id=False):
        """

        :param root_path:
        :param labels_file_name: File path to file containing labels
        :param data_list: VideoRecord item list of train/valid/test split (which list depends on Handler call)
        :param num_segments:
        :param new_length:
        :param modality: Is data RGB or FLOW?
        :param image_tmpl: template for image files, ex.: 'img_{:05d}.jpg'
        :param transform:
        :param normalize:
        :param random_shift:
        :param test_mode:
        :param video_suffix: Suffix for stereo videos, to chose camera image - left or right side
        :param return_3D_tensor:
        :param return_three_channels:
        :param preload_to_RAM: Should data be preloaded to RAM?
        :param return_trial_id:
        """

        self.root_path = root_path
        self.labels_file_name = labels_file_name
        self.num_segments = num_segments
        self.new_length = new_length  # number of consecutive frames contained in a snippet
        self.modality = modality
        self.image_tmpl = image_tmpl
        self.transform = transform
        self.normalize = normalize
        self.random_shift = random_shift
        self.test_mode = test_mode

        self.video_suffix = video_suffix
        self.return_3D_tensor = return_3D_tensor
        self.return_three_channels = return_three_channels
        self.preload_to_RAM = preload_to_RAM
        self.return_trial_id = return_trial_id

        self.videorecord_dataset = data_list

        if self.modality == 'RGBDiff':
            self.new_length += 1  # Diff needs one more image to calculate diff

        if self.preload_to_RAM:
            self._preload_images()

    def _load_image(self, directory, idx):
        if self.modality == 'RGB' or self.modality == 'RGBDiff':
            return [Image.open(os.path.join(directory, self.image_tmpl.format(idx))).convert('RGB')]
        elif self.modality == 'Flow':
            x_img = Image.open(os.path.join(directory, self.image_tmpl.format('x', idx + 1))).convert('L')
            y_img = Image.open(os.path.join(directory, self.image_tmpl.format('y', idx + 1))).convert('L')

            return [x_img, y_img]

    def _preload_images(self):
        self.image_data = {}
        for i, record in enumerate(self.videorecord_dataset):
            print(f"dataset.py: Loading images for {record.trial} ({i}/{len(self.videorecord_dataset)})...")
            images = []
            img_dir = os.path.join(self.root_path, record.trial + self.video_suffix)
            for p in range(0, record.frame_count):
                images.extend(self._load_image(img_dir, p))
            self.image_data[record.trial] = images

    def _sample_indices(self, record):
        """

        :param record: VideoRecord
        :return: list
        """
        average_duration = (record.frame_count - self.new_length + 1) // self.num_segments
        if average_duration > 0:
            offsets = np.multiply(list(range(self.num_segments)), average_duration) + randint(average_duration, size=self.num_segments)
        elif record.frame_count > self.num_segments:
            offsets = np.sort(randint(record.frame_count - self.new_length + 1, size=self.num_segments))
        else:
            offsets = np.zeros((self.num_segments,))
        return offsets

    def _get_val_indices(self, record):
        if record.frame_count > self.num_segments + self.new_length - 1:
            tick = (record.frame_count - self.new_length + 1) / float(self.num_segments)
            offsets = np.array([int(tick / 2.0 + tick * x) for x in range(self.num_segments)])
        else:
            offsets = np.zeros((self.num_segments,))
        return offsets

    def _get_test_indices(self, record):
        tick = (record.frame_count - self.new_length + 1) / float(self.num_segments)
        offsets = np.array([int(tick / 2.0 + tick * x) for x in range(self.num_segments)])
        return offsets

    def __getitem__(self, index):
        record = self.videorecord_dataset[index]

        if not self.test_mode:
            segment_indices = self._sample_indices(record) if self.random_shift else self._get_val_indices(record)
        else:
            segment_indices = self._get_test_indices(record)

        return self.get(record, segment_indices)

    def _get_snippet(self, record, seg_ind):
        snippet = list()
        p = int(seg_ind)
        for _ in range(self.new_length):
            if self.preload_to_RAM:
                if self.modality == 'RGB' or self.modality == 'RGBDiff':
                    seg_imgs = self.image_data[record.trial][p: p + 1]
                elif self.modality == 'Flow':
                    idx = p * 2
                    seg_imgs = self.image_data[record.trial][idx: idx + 2]
            else:
                img_dir = os.path.join(self.root_path, record.trial + self.video_suffix)
                seg_imgs = self._load_image(img_dir, p)
            snippet.extend(seg_imgs)
            if p < (record.frame_count - 1):
                p += 1
        return snippet

    def get(self, record, indices):

        images = list()
        for seg_ind in indices:
            images.extend(self._get_snippet(record, seg_ind))

        if self.return_3D_tensor:
            images = self.transform(images)
            images = [ToTensor()(img) for img in images]
            if self.modality == 'RGB':
                images = torch.stack(images, 0)
            elif self.modality == 'Flow':
                _images = []
                if self.return_three_channels:
                    for i in range(len(images) // 2):
                        image_dummy = (images[i] + images[i + 1]) / 2
                        _images.append(torch.cat([images[i], images[i + 1], image_dummy], 0))
                else:
                    for i in range(len(images) // 2):
                        _images.append(torch.cat([images[i], images[i + 1]], 0))
                images = torch.stack(_images, 0)
            images = self.normalize(images)
            images = images.view(((-1, self.new_length) + images.size()[-3:]))
            images = images.permute(0, 2, 1, 3, 4)
            process_data = images
        else:
            transform = Compose([
                self.transform,
                Stack(roll=False),
                ToTensor(),
                self.normalize,
            ])
            process_data = transform(images)

        target = record.label

        if self.return_trial_id:
            trial_id = record.trial.split('_')[-1]
            return trial_id, process_data, target
        else:
            return process_data, target

    def __len__(self):
        return len(self.videorecord_dataset)


# testing
if __name__ == '__main__':
    train_set = DatasetHandler("/mnt/cluster/datasets/AachSut/frames_10fps", "data/OSATS.xlsx")
    print(len(train_set.train))
