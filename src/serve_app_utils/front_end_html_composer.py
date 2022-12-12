PORT_CONFIG = {
    "PROD": 5000,
    "TEST": 5001,
    "DEV": 5002
}

import sys

def generate_front_end_html(environment):
    """

    :param environment: one of "PROD", "TEST" or "DEV"
    :return:
    """
    base_html = """
    <html>
    <head>
        <title>{} environment</title>
    </head>
    <body>
    <form action="http://localhost:{}/login" method="post">
    <p> PoC: Delay predictions for 2023 assuming 2022 schedule and increased customer demand <p>
    <p> Enter in the box below the expected demand increase in 2023 w.r.t 2022. 
	For example by typing 0.1 it means you expect a 10% increase in 2023 demand w.r.t 2022. <p>
	<p>The algorithm will predict on the 2022 flights but with more passengers, increasing the amount of passengers
	per flight uniformly. However, even if you input a 10% increase, the amount of passengers will only
	increase by a smaller factor, since there are flights that were fully booked in 2022, so in 2023 we would
	just lose those passengers. <p>
	<p>You can also pass different values separated with a space (eg: 0.1 0.25 0.5) if you want to test different scenarios.<p>
    <p>Model metrics during testing can be obtained by typing: model metrics<p>
    <p><input type="text" name="increases" /></p>
    <p><input type="submit" value="Submit" /></p>
    </form>
    </body>
    </html>
    """.format(environment, PORT_CONFIG[environment])

    file_html = open("{}_front_end.html".format(environment), "w")
    # Adding the input data to the HTML file
    file_html.write(base_html)
    # Saving the data into the HTML file
    file_html.close()

if __name__ == "__main__":
    env = sys.argv[1]
    generate_front_end_html(env)