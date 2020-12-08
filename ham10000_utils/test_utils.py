from unittest import TestCase
import tempfile
import os
from ham10000_utils import utils


class Test(TestCase):
    def test_duplicate_file_path_no_counter(self):
        filepath = '/some/path/file.jpg'
        self.assertEqual('/some/path/file-1.jpg', utils._duplicate_file_path(filepath))

    def test_duplicate_file_path_with_counter_1(self):
        filepath = '/some/path/file-1.jpg'
        self.assertEqual('/some/path/file-2.jpg', utils._duplicate_file_path(filepath))

    def test_duplicate_file_path_with_counter_2(self):
        filepath = '/some/path/file-2.jpg'
        self.assertEqual('/some/path/file-3.jpg', utils._duplicate_file_path(filepath))

    def test_check_duplicate_one_iteration(self):
        with tempfile.NamedTemporaryFile() as tmp_f:
            filepath = utils._check_duplicate(tmp_f.name, True)
        self.assertEqual(1, utils._count_from_name(filepath, '-'))

    def test_check_duplicate_two_iterations_non_existing_dup(self):
        with tempfile.NamedTemporaryFile() as tmp_f:
            filepath = utils._check_duplicate(tmp_f.name, True)
        filepath = utils._check_duplicate(filepath, True)
        self.assertEqual(1, utils._count_from_name(filepath, '-'))

    def test_check_duplicate_two_iterations_existing_dup(self):
        with tempfile.NamedTemporaryFile() as tmp_f:
            tmp_file_name = tmp_f.name
            filepath = utils._check_duplicate(tmp_file_name, True)
            # create dup file
            with open(filepath, 'w'):
                file_to_remove_1 = filepath
                filepath = utils._check_duplicate(tmp_file_name, True)
            # create dup file
            with open(filepath, 'w'):
                file_to_remove_2 = filepath
                filepath = utils._check_duplicate(tmp_file_name, True)
        # remove dup file
        os.remove(file_to_remove_1)
        # remove dup file
        os.remove(file_to_remove_2)
        self.assertEqual(3, utils._count_from_name(filepath, '-'))


