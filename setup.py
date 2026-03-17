from setuptools import find_packages, setup

package_name = 'dvrk_shujiro'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='Shujiro',
    maintainer_email='shujiros@stanford.edu',
    description='dVRK data collection and analysis by Shujiro',
    license='MIT',
    extras_require={
        'test': [
            'pytest',
        ],
    },
    entry_points={
        'console_scripts': [
            'task_timer = dvrk_shujiro.task_timer:main',
            'task_timer_bar = dvrk_shujiro.task_timer_bar:main',
            'task_timer_gui = dvrk_shujiro.task_timer_gui:main',
        ],
    },
)

