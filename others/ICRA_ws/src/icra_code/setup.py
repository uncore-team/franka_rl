from setuptools import setup, find_packages

package_name = 'icra_code'

setup(
    name=package_name,
    version='0.0.0',

    #packages=[package_name],
    packages=find_packages(),

    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='diego',
    maintainer_email='dcaruanamontes@gmail.com',
    description='TODO: Package description',
    license='TODO: License declaration',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
                'talker = icra_code.pub:main',
                'goal_pub = icra_code.SEQUENCE:main',
        ],
    },
)
