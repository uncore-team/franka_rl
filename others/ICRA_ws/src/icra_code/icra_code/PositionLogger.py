import polars as pl

# Y si uso un publisher de ROS2 y rosbags?

class PositionLogger(): # Usar .csv

    def __init__(self, file: str):
        self.file = file
        self.df = pl.DataFrame()

    def add_row(self, real_pose, desired_pose, time):
        row = pl.DataFrame(
            {
                "real_pose": [real_pose],
                "desired_pose": [desired_pose],
                "time": [time]
            }
        )
        
        self.df = pl.concat([self.df, row])

    def save(self):
        self.df.write_parquet(self.file)


if __name__=="__main__":
    logger = PositionLogger("my_database.parquet")

    print(logger.df)

    logger.add_row([1, 1, 1], [1, 1, 1], 1)

    print(logger.df)

    logger.add_row([1, 1, 1], [1, 1, 1], 1)
    print(logger.df)

    logger.add_row([1, 1, 1], [1, 1, 1], 1)
    print(logger.df)

    logger.save()

    #logger.df.plot.

