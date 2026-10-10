from glob import glob
from setuptools import setup

package_name = 'cyclops_ros'
setup(
    name=package_name,
    version='1.0.0',
    packages=[package_name],
    data_files=[
        ('share/ament_index/resource_index/packages', ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        ('share/' + package_name + '/launch', glob('launch/*.py')),
        ('share/' + package_name + '/config', glob('config/*')),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='Amit Singh Chauhan',
    description='CYCLOPS monocular obstacle avoidance nodes',
    license='MIT',
    entry_points={'console_scripts': [
        'perception_node = cyclops_ros.perception_node:main',
        'planner_node = cyclops_ros.planner_node:main',
    ]},
)
