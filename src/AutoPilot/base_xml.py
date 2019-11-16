def get_base_xml(out_wall,road,end,wall,start_x,start_y,start_z,bound):
    return f'''
    <?xml version="1.0" encoding="UTF-8" standalone="no" ?>
<Mission xmlns="http://ProjectMalmo.microsoft.com" xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance">

  <About>
    <Summary>Cliff walking mission based on Sutton and Barto.</Summary>
  </About>

  <ServerSection>
              <ServerInitialConditions>
                <Time>
                    <StartTime>1000</StartTime>
                    <AllowPassageOfTime>false</AllowPassageOfTime>
                </Time>
                <Weather>clear</Weather>
              </ServerInitialConditions>
    <ServerHandlers>
      <FlatWorldGenerator generatorString="3;7,2;1;"/>
      <DrawingDecorator>
        <DrawCuboid x1="-100"  y1="2" z1="-100"  x2="100" y2="20" z2="100" type="air" />
        <DrawCuboid x1="-100"  y1="1" z1="-100"  x2="100" y2="1" z2="100" type="grass" />
        {out_wall}
        {road}
        {end}
        {wall}
      </DrawingDecorator>
      <ServerQuitFromTimeUp timeLimitMs="39000"/>
      <ServerQuitWhenAnyAgentFinishes/>
    </ServerHandlers>
  </ServerSection>
    
  <AgentSection mode="Survival">
    <Name>Cristina</Name>
    <AgentStart>
      <Placement x="{start_x}" y="{start_y}" z="{start_z}" pitch="10" yaw="0"/>
    </AgentStart>
    <AgentHandlers>
        <VideoProducer want_depth="true">
                <Width>1024</Width>
                <Height>1024</Height>
            </VideoProducer>
        <ContinuousMovementCommands/>
        <AbsoluteMovementCommands/>
      <RewardForTouchingBlockType>
        <Block reward="-1" type="iron_block" behaviour="constant"/>
        <Block reward="-1.5" type="grass" behaviour="constant"/>
        <Block reward="-1.5" type="dirt" behaviour="constant"/>
      </RewardForTouchingBlockType>
    <RewardForMissionEnd rewardForDeath="0">
        <Reward description="found_goal" reward="250" />
        <Reward description="out" reward="-65" />
    </RewardForMissionEnd>
      <RewardForSendingCommand reward="-1" />
      <AgentQuitFromTouchingBlockType>
          <Block type="redstone_block" description="found_goal"/>
        <Block type="iron_block" description="out"/>
      </AgentQuitFromTouchingBlockType>
      <ObservationFromGrid>
          <Grid name="floorAll">
                        <min x="-10" y="-1" z="-10"/>
                        <max x="10" y="-1" z="10"/>
          </Grid>
      </ObservationFromGrid>
    </AgentHandlers>
    
  </AgentSection>

</Mission>
    '''