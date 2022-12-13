import unittest
import os
from openprotein.core.setup import GlobalConfiguration, Tree
import rtoml

class setupTest(unittest.TestCase):
    def setUp(self):
        self.toml_str = """
            title = "Toml Example"
            
            ["owner"]
            developer = "zju"
            inner.developer = "zj"
            
            ["info"]
            version = 0.1
            state = "release"
        """
        self.other_str = """
            update.info = true
            ["info"]
            state = "developer"
        """
        self.path = "./core/sys.toml"
        os.chdir(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
        # self.args = GlobalConfiguartion(self.path)

    def test_ConfigDict(self):
        cd = Tree()
        x = rtoml.load(self.toml_str)
        cd.update(x)
        y = rtoml.load(self.other_str)
        cd.update(y)
        cd["info"]
        cd.info.version


    def test_GlobalConfiguartion(self):
        args = GlobalConfiguration(self.path)
        args.sys
        args.sys.backend
        pass

    # def test_component

if __name__ == "__main__":
    unittest.main()